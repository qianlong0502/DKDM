from omegaconf import OmegaConf
from guided_diffusion import logger

def check_args(teacher_args, student_args):
    teacher_model_args = OmegaConf.load(teacher_args.model)
    teacher_diffusion_args = OmegaConf.load(teacher_args.diffusion)
    del teacher_args.model
    del teacher_args.diffusion
    teacher_args = OmegaConf.merge(teacher_args, teacher_model_args, teacher_diffusion_args)
    student_model_args = OmegaConf.load(student_args.model)
    student_diffusion_args = OmegaConf.load(student_args.diffusion)
    del student_args.model
    del student_args.diffusion
    student_args = OmegaConf.merge(student_args, student_model_args, student_diffusion_args)

    if student_args.diffusion_steps != teacher_args.diffusion_steps:
        logger.log(f"WARNING: student_args.diffusion_steps ({student_args.diffusion_steps}) != teacher_args.diffusion_steps ({teacher_args.diffusion_steps}), setting student_args.diffusion_steps to {teacher_args.diffusion_steps}")
        student_args.diffusion_steps = teacher_args.diffusion_steps

    # for way of repace
    if teacher_args.timestep_respacing != '' and student_args.timestep_respacing != teacher_args.timestep_respacing:
        logger.log(f"WARNING: student_args.timestep_respacing ({student_args.timestep_respacing}) != teacher_args.timestep_respacing ({teacher_args.timestep_respacing}), setting student_args.timestep_respacing to {teacher_args.timestep_respacing}")
        student_args.timestep_respacing = teacher_args.timestep_respacing

    # student_args.input_perturb = teacher_args.input_perturb

    return teacher_args, student_args