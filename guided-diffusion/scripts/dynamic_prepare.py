"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append('.')
import datetime

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.train_util_kd import KDTeacherLoader

from omegaconf import OmegaConf
from scripts.utils import check_args

def main():
    # args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--exp_dir", type=str, default='experiments/kd', help="experiment directory")
    args = argparser.parse_args()
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    EXP_DIR = args.exp_dir
    NAME = args.config.split('/')[-1].split('.')[0]
    args = OmegaConf.load(args.config)
    teacher_args, student_args = args.teacher, args.student
    teacher_args, student_args = check_args(teacher_args, student_args)

    # cache_dir
    cache_dir = os.path.dirname(teacher_args.tempfile)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # log
    log_dir = "prepare-" + NAME
    log_dir = datetime.datetime.now().strftime(f"{log_dir}-%Y-%m-%d-%H-%M-%S-%f")
    log_dir = os.path.join(os.getcwd(), EXP_DIR, log_dir)
    logger.configure(log_dir)

    # dist
    dist_util.setup_dist()
    rank = dist_util.get_rank()
    world_size = dist_util.get_world_size()

    total_batch_size = student_args.batch_size
    assert total_batch_size % world_size == 0, "batch size should be divisible by world size"
    student_args.batch_size = total_batch_size // world_size

    dist.barrier()

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
    _, student_diffusion = create_model_and_diffusion(
        **args_to_dict(student_args, model_and_diffusion_defaults().keys())
    )

    assert teacher_args.iter_choice == 'dynamic_iterative_distillation'

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
        tempfile=args.teacher.tempfile,
    )

    teacher_loader.generate()

if __name__ == "__main__":
    main()
