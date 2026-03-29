"""
在 TopK 的 indices 输出（int64）后插入 Cast(int64→int64)，
打断 ATC 内部的 TopK+TransposeD 融合，解决 CANN 7.0.RC1 Transpose int32 tiling 报错
"""
import copy
import onnx
from onnx import helper, TensorProto, shape_inference

def break_topk_transpose_fusion(input_path, output_path):
    model = onnx.load(input_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    new_nodes = []
    fix_count = 0

    for node in graph.node:
        new_nodes.append(node)

        if node.op_type == "TopK":
            # TopK output[1] 是 indices（int64），插入一个 Cast(int64→int64)
            indices_out = node.output[1]           # 原始 indices 输出名
            barrier_out = f"__topk_barrier_{fix_count}"  # 新的中间张量名

            # 把后续所有用到 indices_out 的节点，改成用 barrier_out
            for later_node in graph.node:
                for i, inp in enumerate(later_node.input):
                    if inp == indices_out:
                        later_node.input[i] = barrier_out

            # 插入 Cast(int64 → int64)
            cast_node = helper.make_node(
                "Cast",
                inputs=[indices_out],
                outputs=[barrier_out],
                to=int(TensorProto.INT64),
                name=f"__cast_topk_barrier_{fix_count}"
            )
            new_nodes.append(cast_node)
            print(f"  [Fix {fix_count}] Added Cast barrier after TopK '{node.name}'")
            print(f"    indices: {indices_out} -> {barrier_out}")
            fix_count += 1

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"\n共修复 {fix_count} 处 TopK fusion")
    print(f"保存至: {output_path}")

if __name__ == "__main__":
    break_topk_transpose_fusion(
        "F:/code1.0/pig_tracking_system/deploy/inference_model.onnx",
        "F:/code1.0/pig_tracking_system/deploy/inference_model_fixed.onnx"
    )
