"""
Train a diffusion model on images.
"""

import sys
sys.path.append('.')

import argparse
from omegaconf import OmegaConf

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import os
import datetime
import torch.distributed as dist


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--data_dir", type=str, required=True)
    argparser.add_argument("--wandb", action="store_true", help="use wandb")
    argparser.add_argument("--exp_dir", type=str, default='experiments/kd', help="experiment directory")
    args = argparser.parse_args()
    data_dir = args.data_dir
    NAME = args.config.split('/')[-1].split('.')[0]
    WANDB = args.wandb
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    EXP_DIR = args.exp_dir
    args = OmegaConf.load(args.config)
    args.data_dir = data_dir
    model_args = OmegaConf.load(args.model)
    diffusion_args = OmegaConf.load(args.diffusion)
    del args.model
    del args.diffusion
    args = OmegaConf.merge(args, model_args, diffusion_args)

    dist_util.setup_dist()
    rank = dist.get_rank()
    world_size = dist_util.get_world_size()

    # wandb
    if WANDB and rank == 0:
        print('initialize wandb')
        import wandb
        wandb.init(
            project='kd-diffusion',
            entity='ilearn',
            name=NAME,
            config=OmegaConf.to_container(args),
        )

    total_batch_size = args.batch_size
    assert total_batch_size % world_size == 0, "batch size should be divisible by world size"
    args.batch_size = total_batch_size // world_size
    dist.barrier()

    # log
    log_dir = datetime.datetime.now().strftime(f"{NAME}-%Y-%m-%d-%H-%M-%S-%f")
    log_dir = os.path.join(os.getcwd(), EXP_DIR, log_dir)
    logger.configure(log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    w = wandb if (WANDB and rank == 0) else None
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip_last=args.clip_last,
    ).run_loop(w)


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip_last=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
