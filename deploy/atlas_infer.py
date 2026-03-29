"""
pyACL NPU 推理封装 — Atlas 200I DK A2

加载 inference_model.om，对外提供与 RFDETRBase 兼容的 predict() 接口：
    results = infer.predict(bgr_frame, conf_thresh=0.5)
    results.xyxy        # np.ndarray [N, 4]  像素坐标 [x1,y1,x2,y2]
    results.confidence  # np.ndarray [N]     置信度

模型规格（export_onnx.py 导出）：
    输入:  [1, 3, 560, 560] float32 NCHW
    输出0 (dets):   [1, 300, 4]  归一化 cxcywh
    输出1 (labels): [1, 300, C]  原始 logits（需 sigmoid）
"""

import os
import cv2
import numpy as np
from types import SimpleNamespace

# ImageNet 归一化参数
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_H = INPUT_W = 560  # 模型固定输入尺寸

# ACL 内存拷贝方向常量
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0


class AscendInfer:
    """
    Atlas 200I DK A2 NPU 推理类。

    示例:
        infer = AscendInfer("/root/pig_tracking/inference_model.om")
        results = infer.predict(frame)
        # results.xyxy [N,4], results.confidence [N]
        infer.release()
    """

    def __init__(self, model_path: str, device_id: int = 0):
        try:
            import acl
            self._acl = acl
        except ImportError:
            raise ImportError(
                "acl 模块未找到。请确认已激活 CANN 环境：\n"
                "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
            )

        self.device_id = device_id
        self._context = None
        self._model_id = None
        self._model_desc = None
        self._input_dataset = None
        self._output_dataset = None
        self._input_bufs = []
        self._input_sizes = []
        self._output_bufs = []
        self._output_sizes = []
        self._released = False

        self._init_acl()
        self._load_model(model_path)
        self._prepare_buffers()

    # ------------------------------------------------------------------
    # ACL 生命周期
    # ------------------------------------------------------------------

    def _init_acl(self):
        acl = self._acl
        ret = acl.init()
        assert ret == 0, f"acl.init() 失败: {ret}"

        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"set_device({self.device_id}) 失败: {ret}"

        self._context, ret = acl.rt.create_context(self.device_id)
        assert ret == 0, f"create_context 失败: {ret}"

    def _load_model(self, model_path: str):
        acl = self._acl
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"OM 模型不存在: {model_path}")

        self._model_id, ret = acl.mdl.load_from_file(model_path)
        assert ret == 0, f"load_from_file 失败: {ret}"

        self._model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._model_desc, self._model_id)
        assert ret == 0, f"get_desc 失败: {ret}"

        n_in  = acl.mdl.get_num_inputs(self._model_desc)
        n_out = acl.mdl.get_num_outputs(self._model_desc)
        print(f"[AscendInfer] 模型加载成功: {model_path}")
        print(f"[AscendInfer] 输入数量: {n_in}  输出数量: {n_out}")

    def _prepare_buffers(self):
        acl = self._acl
        n_in  = acl.mdl.get_num_inputs(self._model_desc)
        n_out = acl.mdl.get_num_outputs(self._model_desc)

        # ---- 输入 dataset ----
        self._input_dataset = acl.mdl.create_dataset()
        for i in range(n_in):
            size = acl.mdl.get_input_size_by_index(self._model_desc, i)
            buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == 0, f"malloc input[{i}] 失败"
            data_buf = acl.create_data_buffer(buf, size)
            # CANN 7.x pyACL: add_dataset_buffer 返回 (data_buf, ret) 二元组
            result = acl.mdl.add_dataset_buffer(self._input_dataset, data_buf)
            ret = result[1] if isinstance(result, tuple) else result
            assert ret == 0, f"add_dataset_buffer input[{i}] 失败: {ret}"
            self._input_bufs.append(buf)
            self._input_sizes.append(size)

        # ---- 输出 dataset ----
        self._output_dataset = acl.mdl.create_dataset()
        for i in range(n_out):
            size = acl.mdl.get_output_size_by_index(self._model_desc, i)
            buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == 0, f"malloc output[{i}] 失败"
            data_buf = acl.create_data_buffer(buf, size)
            result = acl.mdl.add_dataset_buffer(self._output_dataset, data_buf)
            ret = result[1] if isinstance(result, tuple) else result
            assert ret == 0, f"add_dataset_buffer output[{i}] 失败: {ret}"
            self._output_bufs.append(buf)
            self._output_sizes.append(size)

    # ------------------------------------------------------------------
    # 前处理 / 推理 / 后处理
    # ------------------------------------------------------------------

    def _preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 → NCHW float32 归一化，连续内存"""
        img = cv2.resize(bgr_frame, (INPUT_W, INPUT_H))   # HWC BGR uint8
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # RGB [0,1]
        img = (img - _MEAN) / _STD                         # 归一化
        img = img.transpose(2, 0, 1)[np.newaxis]           # NCHW
        return np.ascontiguousarray(img, dtype=np.float32)

    def _infer_raw(self, input_np: np.ndarray) -> list:
        """拷贝输入→NPU，执行推理，拷贝输出→host，返回 list[np.ndarray(uint8)]"""
        acl = self._acl

        # H2D（使用 numpy ctypes 指针）
        ret = acl.rt.memcpy(
            self._input_bufs[0],
            self._input_sizes[0],
            input_np.ctypes.data,
            self._input_sizes[0],
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        assert ret == 0, f"H2D 拷贝失败: {ret}"

        # 执行推理
        ret = acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset)
        assert ret == 0, f"mdl.execute 失败: {ret}"

        # D2H
        outputs = []
        for buf, size in zip(self._output_bufs, self._output_sizes):
            out = np.empty(size, dtype=np.uint8)
            ret = acl.rt.memcpy(
                out.ctypes.data,
                size,
                buf,
                size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            assert ret == 0, f"D2H 拷贝失败: {ret}"
            outputs.append(out)
        return outputs

    def _postprocess(
        self,
        raw_outputs: list,
        orig_h: int,
        orig_w: int,
        conf_thresh: float,
    ):
        """
        raw_outputs[0] → dets   [1, 300, 4]  归一化 cxcywh
        raw_outputs[1] → labels [1, 300, C]  logits（需 sigmoid）
        返回 (xyxy_pixels [N,4], confidences [N])
        """
        dets   = raw_outputs[0].view(np.float32).reshape(300, 4)   # (300, 4)
        labels = raw_outputs[1].view(np.float32).reshape(300, -1)  # (300, C)

        # 置信度 = 各类 sigmoid 最大值
        scores = 1.0 / (1.0 + np.exp(-np.clip(labels, -88, 88)))  # sigmoid，防溢出
        conf   = scores.max(axis=1)                                  # (300,)

        # 置信度过滤
        mask = conf >= conf_thresh
        if not mask.any():
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = dets[mask]   # (N, 4) cxcywh 归一化 [0,1]
        confs = conf[mask]   # (N,)

        # cxcywh 归一化 → xyxy 像素
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.clip((cx - bw / 2) * orig_w, 0, orig_w)
        y1 = np.clip((cy - bh / 2) * orig_h, 0, orig_h)
        x2 = np.clip((cx + bw / 2) * orig_w, 0, orig_w)
        y2 = np.clip((cy + bh / 2) * orig_h, 0, orig_h)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        return xyxy.astype(np.float32), confs.astype(np.float32)

    def predict(self, bgr_frame: np.ndarray, conf_thresh: float = 0.5) -> SimpleNamespace:
        """
        与 RFDETRBase.predict() 兼容的接口。
        返回 SimpleNamespace(xyxy=[N,4], confidence=[N])
        """
        orig_h, orig_w = bgr_frame.shape[:2]
        inp = self._preprocess(bgr_frame)
        raw = self._infer_raw(inp)
        xyxy, confs = self._postprocess(raw, orig_h, orig_w, conf_thresh)
        return SimpleNamespace(xyxy=xyxy, confidence=confs)

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------

    def release(self):
        if self._released:
            return
        self._released = True
        acl = self._acl

        for buf in self._input_bufs + self._output_bufs:
            acl.rt.free(buf)

        if self._input_dataset is not None:
            acl.mdl.destroy_dataset(self._input_dataset)
        if self._output_dataset is not None:
            acl.mdl.destroy_dataset(self._output_dataset)
        if self._model_desc is not None:
            acl.mdl.destroy_desc(self._model_desc)
        if self._model_id is not None:
            acl.mdl.unload(self._model_id)
        if self._context is not None:
            acl.rt.destroy_context(self._context)

        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[AscendInfer] 资源已释放")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
