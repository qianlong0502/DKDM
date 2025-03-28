import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm .models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config

import torch.distributed as dist
import pickle
from collections import defaultdict

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                _print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        raise NotImplementedError
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            raise NotImplementedError
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    _print(f'Throughput for this batch: {log["throughput"]}')
    return log


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=True,
        action='store_false',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--stu_batch_size",
        type=int,
        nargs="?",
        help="the bs of the student in dkdm",
        default=10
    )
    parser.add_argument(
        "--BATCH",
        type=float,
        nargs="?",
        help="the BATCH for cache",
        default=0.4
    )
    parser.add_argument(
        "--save_to",
        type=str,
        nargs="?",
        default="cache",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        _print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    def _print(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    _print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # resume
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            _print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    # load config
    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    _print(config)

    # load model
    model: LatentDiffusion
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    _print(f"global step: {global_step}")

    # write config out
    sampling_conf = vars(opt)
    _print(sampling_conf)


    # dynamic prepare
    if opt.vanilla_sample:
        _print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        raise NotImplementedError('Currently only vanilla sampling supported.')
        _print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')


    tstart = time.time()
    if model.cond_stage_model is None:
        with model.ema_scope("Plotting"):
            total_t, total_x_t_1 = model.dynamic_prepare(batch_size=opt.batch_size, stu_batch_size=opt.stu_batch_size, BATCH=opt.BATCH, verbose=True)

        result = defaultdict(None)
        result['total_t'] = total_t.cpu().numpy()
        result['total_x_t_1'] = total_x_t_1.cpu().numpy()
        save_path = f"{opt.save_to}_rank{rank}"
        with open(save_path, 'wb') as f:
            print(f'Saving to {save_path}')
            pickle.dump(result, f)
        
        if rank == 0:
            # merge results
            save_paths = [f"{opt.save_to}_rank{r}" for r in range(dist.get_world_size())]
            while True:
                if all([os.path.exists(p) for p in save_paths]):
                    break
                time.sleep(5)
            print('Merging results...')
            final_result = defaultdict(None)
            for p in save_paths:
                with open(p, 'rb') as f:
                    res = pickle.load(f)
                    for k in res:
                        if k not in final_result:
                            final_result[k] = res[k]
                        else:
                            final_result[k] = np.concatenate([final_result[k], res[k]], axis=0)
            with open(opt.save_to, 'wb') as f:
                print(f'Saving to {opt.save_to}')
                pickle.dump(final_result, f)
                print(f'Finished saving to {opt.save_to}')

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')