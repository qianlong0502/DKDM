"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append('.')
import datetime

import argparse
from omegaconf import OmegaConf

import os
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from guided_diffusion.train_util_kd import KDTeacherLoader, KDTrainLoop
from guided_diffusion.resample import create_named_schedule_sampler

from scripts.utils import check_args

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--resume_checkpoint", type=str, default='', help="resume from checkpoint directory")
    argparser.add_argument("--resume_step", type=int, default=0, help="how many times to resume, for config name")
    argparser.add_argument("--wandb", action="store_true", help="use wandb")
    argparser.add_argument("--exp_dir", type=str, default='experiments/kd', help="experiment directory")
    args = argparser.parse_args()
    resume_checkpoint = args.resume_checkpoint
    resume_step = args.resume_step
    NAME = args.config.split('/')[-1].split('.')[0]
    WANDB = args.wandb
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    EXP_DIR = args.exp_dir
    args = OmegaConf.load(args.config)
    if resume_checkpoint: assert resume_step > 0, "resume_step should be greater than 0"
    teacher_args, student_args = args.teacher, args.student
    teacher_args, student_args = check_args(teacher_args, student_args)

    if resume_checkpoint:
        NAME = f"{NAME}-resume-{resume_step}"

    # log
    log_dir = datetime.datetime.now().strftime(f"{NAME}-%Y-%m-%d-%H-%M-%S-%f")
    log_dir = os.path.join(os.getcwd(), EXP_DIR, log_dir)
    logger.configure(log_dir)

    # dist
    dist_util.setup_dist()
    rank = dist_util.get_rank()
    world_size = dist_util.get_world_size()

    # wandb
    if WANDB and rank == 0:
        print('initialize wandb')
        import wandb
        wandb.init(
            project='kd-diffusion',
            entity='ilearn',
            name=NAME,
            config={
                "args": OmegaConf.to_container(args),
                "teacher_args": OmegaConf.to_container(teacher_args),
                "student_args": OmegaConf.to_container(student_args),
            },
        )
    w = wandb if (WANDB and rank == 0) else None

    total_batch_size = student_args.batch_size
    assert total_batch_size % world_size == 0, "batch size should be divisible by world size"
    student_args.batch_size = total_batch_size // world_size

    dist.barrier()

    # log args
    logger.log("=" * 80)
    logger.log("args")
    logger.log("-" * 80)
    logger.log(OmegaConf.to_yaml(args))
    logger.log("=" * 80)
    logger.log("teacher args")
    logger.log("-" * 80)
    logger.log(OmegaConf.to_yaml(teacher_args))
    logger.log("=" * 80)
    logger.log("student args")
    logger.log("-" * 80)
    logger.log(OmegaConf.to_yaml(student_args))
    logger.log("=" * 80)
    logger.log("rank: ", rank)
    logger.log("world_size: ", world_size)

    logger.log("creating model and diffusion of teacher...")
    teacher_args_dict = args_to_dict(teacher_args, model_and_diffusion_defaults().keys())
    teacher_args_dict['kd'] = True
    teacher_model, teacher_diffusion = create_model_and_diffusion(**teacher_args_dict)
    teacher_model.load_state_dict(
        dist_util.load_state_dict(teacher_args.model_path, map_location="cpu"), strict=False
    )
    teacher_model.to(dist_util.dev())
    if teacher_args.use_fp16:
        teacher_model.convert_to_fp16()
    teacher_model.eval()

    logger.log("creating model and diffusion of student...")
    student_model, student_diffusion = create_model_and_diffusion(
        **args_to_dict(student_args, model_and_diffusion_defaults().keys())
    )
    if student_args.model_path and os.path.exists(student_args.model_path) and os.path.isfile(student_args.model_path):
        student_model.load_state_dict(
            dist_util.load_state_dict(student_args.model_path, map_location="cpu")
        )
        logger.log(f"warning, loaded model from {student_args.model_path}, make sure this initialization is what you want.")
    student_model.to(dist_util.dev())

    logger.log("kd training...")

    teacher_loader = KDTeacherLoader(
        teacher_diffusion=teacher_diffusion,
        teacher_model=teacher_model,
        student_diffusion=student_diffusion,
        class_cond=teacher_args.class_cond,
        batch=student_args.batch_size,
        microbatch=student_args.microbatch,
        teacher_batch=teacher_args.batch_size,
        teacher_image_size=teacher_args.image_size,
        clip_denoised=teacher_args.clip_denoised,
        choice=teacher_args.iter_choice,
        prob=teacher_args.prob,
        BATCH=teacher_args.BATCH,
        rank=rank,
        world_size=world_size,
        tempfile=teacher_args.tempfile,
        resume_checkpoint=resume_checkpoint,
    )

    schedule_sampler = create_named_schedule_sampler(student_args.schedule_sampler, student_diffusion)
    trainer = KDTrainLoop(
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        teacher_batch_size=teacher_args.batch_size,
        model=student_model,
        diffusion=student_diffusion,
        class_cond=teacher_args.class_cond,
        clip_denoised=teacher_args.clip_denoised,
        teacher_image_size=teacher_args.image_size,
        student_image_size=student_args.image_size,
        batch_size=student_args.batch_size,
        microbatch=student_args.microbatch,
        lr=student_args.lr,
        ema_rate=student_args.ema_rate,
        log_interval=student_args.log_interval,
        save_interval=student_args.save_interval,
        cache_save_interval=student_args.cache_save_interval,
        resume_checkpoint=resume_checkpoint,
        resume_step=resume_step,
        use_fp16=student_args.use_fp16,
        fp16_scale_growth=student_args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=student_args.weight_decay,
        lr_anneal_steps=student_args.lr_anneal_steps,
        # student_clip_denoised = student_args.clip_denoised,
        # clip_last=student_args.clip_last,
        teacher_loader=teacher_loader,
        input_perturb=teacher_args.input_perturb,
    )

    # thops
    import thop
    x = th.randn(1, 3, student_args.image_size, student_args.image_size).to(dist_util.dev())
    t = th.randint(0, teacher_args.diffusion_steps, (1,), device=dist_util.dev())
    inputs = (x, t)
    flops, params = thop.profile(student_model, inputs=inputs, verbose=False)
    logger.log(f"Student Parameters: {sum(p.numel() for p in student_model.parameters()):,} (p.numel)")
    logger.log(f"Student Parameters: {params:,} (thop)")
    logger.log(f"Student FLOPs: {flops:,}")

    trainer.run_loop(w)

if __name__ == "__main__":
    main()
