import torch as th
from guided_diffusion.unet import UNetModel
from guided_diffusion.respace import SpacedDiffusion

def denoise_fn(unet: UNetModel,
            x_t: th.Tensor,
            t: th.Tensor,
            teacher_diffusion: SpacedDiffusion,
            clip_denoised: bool,
            model_kwargs: dict,
    ):
    x_t_1 = x_t

    with th.no_grad():
        out = teacher_diffusion.p_mean_variance(
            unet,
            x_t,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x_t)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when total_t == 0
        x_t_1 = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

    return x_t_1, out

def manage_denoise_fn(
        unet: UNetModel,
        total_x_t_1: th.Tensor,
        total_t: th.Tensor,
        total_t_end: th.Tensor,
        teacher_batch: int,
        clip_denoised: bool,
        teacher_diffusion: SpacedDiffusion,
        class_cond=False,
        total_classes=None,
        pbar=None
    ):
    while (total_t > total_t_end).any():
        total_denoised_index = th.where(total_t > total_t_end)[0]
        if len(total_denoised_index) == 0: break

        denoise_index = total_denoised_index[:teacher_batch]
        total_t[denoise_index] = total_t[denoise_index] - 1
        model_kwargs = {}
        if class_cond:
            raise NotImplementedError
            classes = total_classes[denoise_index]#.contiguous()
            model_kwargs = {'y': classes}
        t, x_t_1 = total_t[denoise_index], total_x_t_1[denoise_index]
        x_t = x_t_1
        
        # update cache
        total_x_t_1[denoise_index], _ = denoise_fn(unet, x_t, t, teacher_diffusion, clip_denoised, model_kwargs)
        
        if pbar is not None:
            pbar.update(len(denoise_index))

    return total_x_t_1, total_t