import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import datetime
import random
import string
import math
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .script_util import NUM_CLASSES
from .gaussian_diffusion import _extract_into_tensor, ModelMeanType
from .denoise_fn import manage_denoise_fn, denoise_fn

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        clip_last=False,
        USE_LOG=True
    ):
        self.clip_last = clip_last
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.USE_LOG = USE_LOG

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self, w=None):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                now = datetime.datetime.now()
                now_time = now.strftime("%m-%d-%H-%M-%S")
                logger.log(f"now time: {now_time}")
                logger.dumpkvs(w)
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                clip_last=self.clip_last
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        if self.USE_LOG == False: return
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


class KDTeacherLoader():
    def __init__(self, 
            teacher_diffusion,
            teacher_model,
            student_diffusion,
            class_cond,
            batch,
            microbatch,
            teacher_batch,
            teacher_image_size,
            clip_denoised,
            rank,
            world_size,
            tempfile,
            resume_checkpoint='',
            choice='dynamic_iterative_distillation',
            prob=1.,
            BATCH=0.05,
            resume=None,
        ):
        self.teacher_diffusion = teacher_diffusion
        self.indices = list(range(self.teacher_diffusion.num_timesteps))[::-1]
        self.teacher_model = teacher_model
        self.student_diffusion = student_diffusion
        self.class_cond = class_cond
        self.batch = batch
        self.microbatch = microbatch
        self.teacher_batch = teacher_batch
        self.teacher_image_size = teacher_image_size
        self.clip_denoised = clip_denoised
        self.choice = choice
        self.probability = prob
        self.BATCH = BATCH
        self.rank = rank
        self.world_size = world_size

        self.timestep_map = {k: self.teacher_diffusion.timestep_map[k] for k in range(len(self.teacher_diffusion.timestep_map))}
        self.timestep_map = {k: k for k in range(len(self.timestep_map))}
        # dict to list
        timestep_map = [-1] * (max(self.timestep_map.keys())+1)
        for k, v in self.timestep_map.items():
            timestep_map[k] = v
        self.timestep_map = timestep_map

        self.resume = resume

        if self.choice != 'iterative_distillation':
            self.total_t = None
            self.total_x_t_1 = None
            if self.class_cond:
                raise NotImplementedError
                self.total_classes = None
        self.tempfile = tempfile
        if resume_checkpoint:
            resume_step = parse_resume_step_from_filename(resume_checkpoint)
            cache_checkpoint = bf.join(
                bf.dirname(resume_checkpoint), f"teacher-state-{resume_step:06}.cache"
            )
            if bf.exists(cache_checkpoint):
                logger.log(f"loading cache state from checkpoint: {cache_checkpoint}")
                self.tempfile = cache_checkpoint


    @property
    def iter(self):
        assert self.choice in [
            'iterative_distillation',
            'shffled_iterative_distillation',
            'dynamic_iterative_distillation'
        ], f"choice ({self.choice}) must be one of ['iterative_distillation', 'shffled_iterative_distillation', 'dynamic_iterative_distillation']"
        while True:
            if self.choice == 'iterative_distillation':
                yield from self.iterative_distillation
            else:
                yield from self.dynamic_iterative_distillation
            
    @property
    def iterative_distillation(self, denoise_fn=denoise_fn):
        while True:
            model_kwargs = {}
            if self.class_cond:
                raise NotImplementedError
                classes = th.randint(low=0, high=NUM_CLASSES, size=(self.batch,), device=dist_util.dev())
                model_kwargs["y"] = classes
            shape = (self.batch, 3, self.teacher_image_size, self.teacher_image_size)

            x_t_1 = th.randn(*shape, device=dist_util.dev())
            indices = list(range(self.teacher_diffusion.num_timesteps))[::-1]
            for i in indices:
                x_t = x_t_1
                t = th.tensor([i] * shape[0], device=dist_util.dev())
                x_t_1, out = denoise_fn(self.teacher_model, x_t, t, self.teacher_diffusion, self.clip_denoised, model_kwargs)
                # TODO: here we do not take classifier into account
                _t = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t]

                # Random Discard
                if random.random() < self.probability:
                    yield (x_t_1, x_t, _t, model_kwargs, out)

    @property
    def dynamic_iterative_distillation(self, denoise_fn=denoise_fn):
        indices = list(range(self.teacher_diffusion.num_timesteps))[::-1]
        if self.choice == 'dynamic_iterative_distillation':
            total_batch_size = round(self.BATCH * self.student_diffusion.num_timesteps) * self.batch
        else:
            total_batch_size = self.batch
        if self.resume is not None and self.resume != '':
            self.tempfile = '...'
            raise NotImplementedError
            TEMP = os.environ.get('TEMP', None)
            if TEMP is None:
                tempfile = f'{self.resume}/{self.teacher_batch}-{self.batch}-{self.BATCH}-{dist_util.get_rank()}-{dist_util.get_world_size()}.pkl'
            else:
                tempfile = f'{self.resume}/{TEMP}-{self.teacher_batch}-{self.batch}-{self.BATCH}-{dist_util.get_rank()}-{dist_util.get_world_size()}.pkl'
            assert os.path.exists(tempfile)
            logger.log(f"loading teacher data from {tempfile}")

        assert os.path.exists(self.tempfile)
        logger.log(f"Loading Dyanmic Data from: {self.tempfile}")
        with open(self.tempfile, 'rb') as f:
            start, end = self.rank * total_batch_size, (self.rank + 1) * total_batch_size
            res = pickle.load(f)
            self.total_t = res['total_t'][start:end]
            assert len(self.total_t) == total_batch_size
            self.total_t = th.tensor(self.total_t, device=dist_util.dev())
            self.total_x_t_1 = res['total_x_t_1'][start:end]
            assert len(self.total_x_t_1) == total_batch_size
            self.total_x_t_1 = th.tensor(self.total_x_t_1, device=dist_util.dev())
            if self.class_cond:
                raise NotImplementedError
        
        logger.log("Dynamic Data loaded.")
        dist.barrier()

        valid_indices = list(range(total_batch_size))
        if self.choice == 'dynamic_iterative_distillation':
            random_index = random.sample(valid_indices, self.batch)
        else:
            random_index = valid_indices
        t, x_t_1 = self.total_t[random_index], self.total_x_t_1[random_index]

        total_step = 0
        while True:
            # check t == 0
            zero_index = th.where(t == 0)[0]

            if len(zero_index) > 0:
                if self.class_cond:
                    raise NotImplementedError
                    self.total_classes[zero_index] = th.randint(low=0, high=NUM_CLASSES, size=(len(zero_index),), device=dist_util.dev())
                t[zero_index] = th.tensor([indices[0]+1] * len(zero_index), device=dist_util.dev())
                _shape = (len(zero_index), 3, self.teacher_image_size, self.teacher_image_size)
                x_t_1[zero_index] = th.randn(*_shape, device=dist_util.dev())

            t = t - 1
            x_t = x_t_1
            assert (t >= 0).all(), f"t: {t}"
            assert (t <= indices[0]).all(), f"indices[0]: {indices[0]}, t: {t}"
            model_kwargs = {}
            if self.class_cond:
                raise NotImplementedError
                classes = self.total_classes[random_index]#.contiguous()
                model_kwargs = {'y': classes}
            x_t_1, out = denoise_fn(self.teacher_model, x_t, t, self.teacher_diffusion, self.clip_denoised, model_kwargs)

            if random.random() < self.probability:
                _t = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t]
                yield (x_t_1, x_t.contiguous(), _t, model_kwargs, out)
                total_step += 1

                # sample new batch
                self.total_t[random_index], self.total_x_t_1[random_index] = t, x_t_1

                if self.choice == 'dynamic_iterative_distillation':
                    random_index = random.sample(valid_indices, self.batch)
                else:
                    random_index = valid_indices

                t, x_t_1 = self.total_t[random_index], self.total_x_t_1[random_index]

    def generate(self, manage_denoise_fn=manage_denoise_fn):
        logger.log("Preparing teacher data...")

        indices = list(range(self.teacher_diffusion.num_timesteps))[::-1]

        total_batch_size = round(self.BATCH * self.student_diffusion.num_timesteps) * self.batch

        if self.rank == 0:
            logger.log(f"total samples: {total_batch_size * self.world_size}")
        
        shape = (total_batch_size, 3, self.teacher_image_size, self.teacher_image_size)

        logger.log(f"Generating teacher data...")

        self.total_classes = None
        if self.class_cond:
            raise NotImplementedError
            self.total_classes = th.randint(low=0, high=NUM_CLASSES, size=(total_batch_size,), device=dist_util.dev())
        
        self.total_t = th.tensor([indices[0]+1] * total_batch_size, device=dist_util.dev())
        self.total_x_t_1 = th.randn(*shape, device=dist_util.dev())

        # For a given sample, it needs to reach this state:
        # At most denoised to (x_t_0)
        # Because x_t_0 can be denoised once more to get the real image
        t_end_candidates = set(indices[:-1])
        total_t_end = random.choices(list(t_end_candidates), k=total_batch_size)
        total_t_end = th.tensor(total_t_end, device=dist_util.dev())

        total_inference_times = (self.total_t - total_t_end).sum().item()

        # multi GPU is not recommended
        # multiple GPU: unify total_inference_times
        if self.world_size > 1:
            logger.log(f"Warning: Using multiple GPUs is not recommended.")
            gather_inference_times = th.tensor([total_inference_times], device=dist_util.dev())
            gather_inference_times_s = [th.zeros(1, device=dist_util.dev(), dtype=gather_inference_times.dtype) for _ in range(self.world_size)]
            dist.all_gather(gather_inference_times_s, gather_inference_times)
            gather_inference_times = th.cat(gather_inference_times_s, dim=0)
            minimum_inference_times = gather_inference_times.min().item()

            while total_inference_times > minimum_inference_times:
                able_to_plus_1 = th.where(total_t_end < indices[0])[0]
                if total_inference_times - minimum_inference_times > len(able_to_plus_1):
                    total_t_end[able_to_plus_1] = total_t_end[able_to_plus_1] + 1
                else:
                    index_to_plus_1 = random.sample(able_to_plus_1.tolist(), total_inference_times - minimum_inference_times)
                    total_t_end[index_to_plus_1] = total_t_end[index_to_plus_1] + 1
                    
                total_inference_times = (self.total_t - total_t_end).sum().item()
            assert total_inference_times == minimum_inference_times

        pbar = tqdm(total=total_inference_times, desc="Generating teacher data", ncols=80) if self.rank == 0 else None

        dist.barrier()

        self.total_x_t_1, self.total_t = manage_denoise_fn(self.teacher_model, self.total_x_t_1, self.total_t, total_t_end, self.teacher_batch, self.clip_denoised, self.teacher_diffusion, self.class_cond, self.total_classes, pbar=pbar)

        dist.barrier()
        if self.rank == 0:
            pbar.close()
            logger.log(f"before gather: total_t: {self.total_t.shape}, total_x_t_1: {self.total_x_t_1.shape}")
        # gather
        total_t_s = [th.zeros_like(self.total_t) for _ in range(self.world_size)]
        dist.all_gather(total_t_s, self.total_t)
        self.total_t = th.cat(total_t_s, dim=0)
        total_x_t_1_s = [th.zeros_like(self.total_x_t_1) for _ in range(self.world_size)]
        dist.all_gather(total_x_t_1_s, self.total_x_t_1)
        self.total_x_t_1 = th.cat(total_x_t_1_s, dim=0)
        if self.class_cond:
            raise NotImplementedError
            total_classes_s = [th.zeros_like(self.total_classes) for _ in range(self.world_size)]
            dist.all_gather(total_classes_s, self.total_classes)
            self.total_classes = th.cat(total_classes_s, dim=0)

        if self.rank == 0:
            logger.log(f"after gather: total_t: {self.total_t.shape}, total_x_t_1: {self.total_x_t_1.shape}")

            with open(self.tempfile, 'wb') as f:
                res = defaultdict(None)
                res['total_t'] = self.total_t.cpu().numpy()
                res['total_x_t_1'] = self.total_x_t_1.cpu().numpy()
                if self.class_cond:
                    raise NotImplementedError
                    res['total_classes'] = self.total_classes.cpu().numpy()
                print(f"Saving teacher data to {self.tempfile}...")
                pickle.dump(res, f)
                print(f"Save done.")

    def save(self, path):
        # gather
        total_t_s = [th.zeros_like(self.total_t) for _ in range(self.world_size)]
        dist.all_gather(total_t_s, self.total_t)
        total_t_s = th.cat(total_t_s, dim=0)
        total_x_t_1_s = [th.zeros_like(self.total_x_t_1) for _ in range(self.world_size)]
        dist.all_gather(total_x_t_1_s, self.total_x_t_1)
        total_x_t_1_s = th.cat(total_x_t_1_s, dim=0)
        if self.class_cond:
            raise NotImplementedError
            total_classes_s = [th.zeros_like(self.total_classes) for _ in range(self.world_size)]
            dist.all_gather(total_classes_s, self.total_classes)
            total_classes_s = th.cat(total_classes_s, dim=0)
        if self.rank == 0:
            with open(path, 'wb') as f:
                res = defaultdict(None)
                res['total_t'] = total_t_s.cpu().numpy()
                res['total_x_t_1'] = total_x_t_1_s.cpu().numpy()
                if self.class_cond:
                    raise NotImplementedError
                print(f"Saving teacher data to {path}...")
                pickle.dump(res, f)
                print(f"Save done.")


class KDTrainLoop(TrainLoop):
    def __init__(self,
                 teacher_model,
                 teacher_diffusion,
                 teacher_batch_size,
                 class_cond,
                 clip_denoised,
                 teacher_image_size,
                 student_image_size,
                 teacher_loader: KDTeacherLoader,
                 clip_last=False,
                 student_clip_denoised=False,
                 cache_save_interval=-1,
                 USE_LOG=True,
                 input_perturb=0.,
                 *args,
                 **kwargs):
        kwargs['data'] = None
        super().__init__(*args, **kwargs)
        self.teacher_diffusion = teacher_diffusion
        self.teacher_batch_size = teacher_batch_size
        self.class_cond = class_cond
        self.clip_denoised = clip_denoised
        self.teacher_image_size = teacher_image_size
        self.student_image_size = student_image_size
        self.teacher_model = teacher_model
        self.clip_last = clip_last
        self.student_clip_denoised = student_clip_denoised
        self.cache_save_interval = cache_save_interval
        if self.cache_save_interval != -1: assert self.cache_save_interval % self.save_interval == 0, f"cache_save_interval % save_interval = {self.cache_save_interval} % {self.save_interval} != 0"
        if self.diffusion.loss_type != self.teacher_diffusion.loss_type:
            logger.log(f"Attention, your teacher and student have different loss type, teacher: {self.teacher_diffusion.loss_type}, student: {self.diffusion.loss_type}")
        if self.diffusion.model_mean_type != self.teacher_diffusion.model_mean_type:
            logger.log(f"Attention, your teacher and student have different model mean type, teacher: {self.teacher_diffusion.model_mean_type}, student: {self.diffusion.model_mean_type}")
        self.teacher_loader = teacher_loader
        self.USE_LOG = USE_LOG
        self.input_perturb = input_perturb

    def run_loop(self, w=None):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            for x_t_1, x_t, _t, model_kwargs, out in self.teacher_loader.iter:
                assert -1 not in _t, f"-1 in _t, _t: {_t}"
                self.run_step(x_t_1, x_t, _t, model_kwargs, out)
                if self.step % self.log_interval == 0:
                    now = datetime.datetime.now()
                    now_time = now.strftime("%m-%d-%H-%M-%S")
                    logger.log(f"now time: {now_time}")
                    logger.dumpkvs(w)
                if self.step % self.save_interval == 0:
                    cache_flag = self.step % self.cache_save_interval == 0 and self.teacher_loader.choice == 'dynamic_iterative_distillation'
                    self.save(cache_flag)
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(self.teacher_loader.choice == 'dynamic_iterative_distillation')

    def run_step(self, x_t_1, x_t, t, model_kwargs, out):
        self.forward_backward(x_t_1, x_t, t, model_kwargs, out)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    # def forward_backward(self, batch, cond):
    def forward_backward(self, x_t_1, x_t, t, model_kwargs, out):
        self.mp_trainer.zero_grad()
        for i in range(0, x_t.shape[0], self.microbatch):
            micro_x_t_1 = x_t_1[i : i + self.microbatch].to(dist_util.dev())
            micro_x_t = x_t[i : i + self.microbatch].to(dist_util.dev())
            micro_t = t[i : i + self.microbatch].to(dist_util.dev())
            micro_out = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in out.items()
            }
            micro_model_kwargs = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in model_kwargs.items()
            }
            last_batch = (i + self.microbatch) >= x_t.shape[0]
            t, weights = self.schedule_sampler.sample(x_t.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_kd,
                self.ddp_model,
                micro_x_t_1,
                micro_x_t,
                micro_t,
                micro_out,
                model_kwargs=micro_model_kwargs,
                clip_denoised=self.student_clip_denoised,
                clip_last=self.clip_last,
                input_perturb=self.input_perturb
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, losses
            )
            self.mp_trainer.backward(loss)

    def log_step(self):
        super().log_step()

    def save(self, cache_flag):
        if self.USE_LOG == False: return
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        if cache_flag:
            teacher_state_path = os.path.join(get_blob_logdir(), f"teacher-state-{(self.step+self.resume_step):06d}.cache")
            logger.log(f"Saving cache to {teacher_state_path}")
            self.teacher_loader.save(teacher_state_path)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, q=4):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(q * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
