import torch
from collections import OrderedDict
from pathlib import Path

ckpt_path = Path("output/resnet50_checkpoint_22320.pt")
out_path = Path("output/resnet50_120.pth")

if not ckpt_path.exists():
    raise FileNotFoundError(f"找不到 checkpoint 文件: {ckpt_path.resolve()}")

ckpt = torch.load(str(ckpt_path), map_location="cpu")

print("checkpoint type:", type(ckpt))

if isinstance(ckpt, dict):
    print("checkpoint keys:", ckpt.keys())

if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    new_state_dict[k] = v

torch.save(new_state_dict, str(out_path))

print("已保存纯模型权重到:")
print(out_path.resolve())
