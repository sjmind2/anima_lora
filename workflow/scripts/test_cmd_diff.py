import sys, tomllib
sys.path.insert(0, "O:/loratool/anima_lora_fork")
from workflow.stages.train import TrainExecutor
from pathlib import Path

def print_cmd(toml_path, stage_id):
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    stage_dir = Path(f"O:/loratool/anima_lora_fork/workflows/hanechan-lokr-two-stage/runs/20260529-080401/{stage_id}")
    exec = TrainExecutor(stage_id, config, stage_dir, {})
    resolved = exec.prepare_config({})
    dataset_path = stage_dir / "dataset_config.toml"
    cmd = exec._build_train_cmd(resolved, dataset_path)
    cmd += ["--output_dir", str(stage_dir / "output")]
    return cmd

print("=== TRAIN_S1 CMD ===")
cmd1 = print_cmd("O:/loratool/anima_lora_fork/workflows/hanechan-lokr-two-stage/configs/train_s1.toml", "train_s1")
for p in cmd1:
    print(f"  {p}")

print("\n=== TRAIN_S2 CMD ===")
cmd2 = print_cmd("O:/loratool/anima_lora_fork/workflows/hanechan-lokr-two-stage/configs/train_s2.toml", "train_s2")
for p in cmd2:
    print(f"  {p}")

print("\n=== DIFF ===")
s1 = set(cmd1)
s2 = set(cmd2)
print("In S2 but not S1:")
for p in sorted(s2 - s1):
    print(f"  + {p}")
print("In S1 but not S2:")
for p in sorted(s1 - s2):
    print(f"  - {p}")
