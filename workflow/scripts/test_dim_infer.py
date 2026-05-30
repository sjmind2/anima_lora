import sys
sys.path.insert(0, "O:/loratool/anima_lora_fork")
from safetensors.torch import load_file

ckpt_path = "O:/loratool/anima_lora_fork/workflows/hanechan-lokr-two-stage/runs/20260529-080401/train_s1/output/anima_lokr_s1.safetensors"
weights_sd = load_file(ckpt_path)

modules_dim = {}
modules_alpha = {}
has_lycoris_lokr = False

for key, value in weights_sd.items():
    lora_name = key.rsplit(".", 1)[0]
    
    if "alpha" in key:
        modules_alpha[lora_name] = value
    if key.endswith(".lokr_w1") or key.endswith(".lokr_w1_a"):
        has_lycoris_lokr = True
    if key.endswith(".lokr_w1") and value.dim() == 2:
        modules_dim[lora_name] = value.size(0)
    elif key.endswith(".lokr_w2_a") and value.dim() == 2:
        modules_dim[lora_name] = value.size(1)

print(f"has_lycoris_lokr: {has_lycoris_lokr}")
print(f"modules_dim count: {len(modules_dim)}")
print(f"modules_alpha count: {len(modules_alpha)}")

if modules_dim:
    first_key = list(modules_dim.keys())[0]
    print(f"Sample module: {first_key}, dim={modules_dim[first_key]}")
