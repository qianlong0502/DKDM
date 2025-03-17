"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append('.')
import datetime

import argparse
import os
import time

from omegaconf import OmegaConf

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
from guided_diffusion.script_util import str2bool    


def main():
    # args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--num_samples", type=int, default=50000)
    argparser.add_argument("--timestep_respacing", type=str, default='50')
    argparser.add_argument("--use_ddim", type=str2bool, default=False)
    argparser.add_argument("--name", type=str, default='')
    args = argparser.parse_args()
    conf = OmegaConf.load(args.config)
    
    log_dir = datetime.datetime.now().strftime(f"Teacher-%Y-%m-%d-%H-%M-%S-%f")
    logger.configure(os.path.join(os.getcwd(), 'experiments', 'sample', log_dir))

    if 'student' in conf.keys():
        conf = conf.student
    conf_model_args = OmegaConf.load(conf.model)
    conf_diffusion_args = OmegaConf.load(conf.diffusion)
    del conf.model
    del conf.diffusion
    conf = OmegaConf.merge(conf, conf_model_args, conf_diffusion_args)
    args = OmegaConf.merge(conf, args.__dict__)

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if os.path.exists(args.model_path) and os.path.isfile(args.model_path):
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
        )
    else:
        logger.log(f"Model not loaded, for time measurement only")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        # import 
        start_time = time.time()
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        end_time = time.time()
        logger.log(f"sample time: {end_time - start_time}")
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        config="",
        model_path="",
        batch_size=2500,
        num_sample=50000,
        timestep_respacing=50,
        use_ddim=False,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
