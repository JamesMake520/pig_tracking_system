"""
修复 ONNX 中 TopK 的 k 输入类型：
CANN 7.0.RC1 不支持 INT64 的 k 输入，将其改为 INT32
同时保留之前插入的 Cast barrier（打断 TopK+Transpose 融合）
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference

def fix_topk_k_dtype(input_path, output_path):
    model = onnx.load(input_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    # 建立 initializer 字典
    init_map = {init.name: init for init in graph.initializer}

    fix_count = 0

    for node in graph.node:
        if node.op_type != "TopK":
            continue

        if len(node.input) < 2:
            print(f"  [Skip] TopK '{node.name}' has no k input")
            continue

        k_input_name = node.input[1]
        print(f"TopK '{node.name}': k input = '{k_input_name}'")

        # Case 1：k 是 initializer（常量权重）
        if k_input_name in init_map:
            init = init_map[k_input_name]
            if init.data_type == TensorProto.INT64:
                k_val = numpy_helper.to_array(init).item()
                print(f"  k value = {k_val} (INT64 → INT32)")

                # 替换为 INT32 initializer
                new_init = numpy_helper.from_array(
                    np.array([k_val], dtype=np.int32),
                    name=k_input_name
                )
                graph.initializer.remove(init)
                graph.initializer.append(new_init)
                fix_count += 1
            else:
                print(f"  k dtype already {init.data_type}, skip")
            continue

        # Case 2：k 是 Constant 节点输出
        for prev_node in graph.node:
            if prev_node.op_type == "Constant" and \
               len(prev_node.output) > 0 and \
               prev_node.output[0] == k_input_name:
                attr = prev_node.attribute[0]
                if attr.HasField("t") and attr.t.data_type == TensorProto.INT64:
                    k_val = numpy_helper.to_array(attr.t).item()
                    print(f"  k value = {k_val} (Constant INT64 → INT32)")
                    # 修改 Constant 节点的 tensor 为 INT32
                    new_tensor = numpy_helper.from_array(
                        np.array([k_val], dtype=np.int32)
                    )
                    attr.t.CopyFrom(new_tensor)
                    fix_count += 1
                break

    if fix_count == 0:
        print("未发现需要修复的 TopK k 输入")
    else:
        print(f"\n共修复 {fix_count} 处 TopK k 类型")

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"保存至: {output_path}")


if __name__ == "__main__":
    # 在已有 barrier fix 的基础上继续修复
    fix_topk_k_dtype(
        "F:/code1.0/pig_tracking_system/deploy/inference_model_fixed.onnx",
        "F:/code1.0/pig_tracking_system/deploy/inference_model_fixed2.onnx"
    )
