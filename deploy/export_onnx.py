"""
直接导出 RF-DETR 为 ONNX，绕过 onnx_graphsurgeon 依赖
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "OC_SORT"))

from rfdetr import RFDETRBase

CHECKPOINT = PROJECT_ROOT / "models" / "checkpoint_best_ema.pth"
OUTPUT_DIR = PROJECT_ROOT / "deploy"
OPSET = 16
RESOLUTION = 560  # 必须是 14 的倍数

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = str(OUTPUT_DIR / "inference_model.onnx")

    print("Loading model...")
    detector = RFDETRBase(pretrain_weights=str(CHECKPOINT))
    # detector.model 是 rfdetr.main.Model 包装器，.model 才是实际 PyTorch 模型
    model = deepcopy(detector.model.model)
    model = model.cpu()
    model.eval()

    print(f"Preparing export mode (resolution={RESOLUTION})...")
    if hasattr(model, "export"):
        model.export()
    # 递归触发各子模块的 export 模式
    for name, m in model.named_modules():
        if hasattr(m, "_export") and not m._export and hasattr(m, "export") and callable(m.export):
            try:
                m.export()
            except Exception:
                pass

    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)

    print(f"Running torch.onnx.export (opset={OPSET})...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            input_names=["input"],
            output_names=["dets", "labels"],
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            opset_version=OPSET,
            dynamic_axes=None,
            verbose=False,
        )

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n导出成功: {output_file}")
    print(f"文件大小: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
