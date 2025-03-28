import torch
model = torch.load("model.ckpt")

state_dict = model["state_dict"]

for key in list(state_dict.keys()):
    if "first_stage_model." not in key:
        del state_dict[key]
    else:
        new_key = key.replace("first_stage_model.", "")
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

ckpt = {
    "state_dict": state_dict
}
torch.save(ckpt, "ae.ckpt")
