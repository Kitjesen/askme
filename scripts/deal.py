import onnx
from onnx import shape_inference

def fix_onnx_dynamic_batch(input_model_path, output_model_path, batch_size=1):
    # 1. 加载 ONNX 模型
    model = onnx.load(input_model_path)
    graph = model.graph

    print(f"正在处理模型: {input_model_path}")

    # 2. 修改输入维度
    for input_tensor in graph.input:
        shape = input_tensor.type.tensor_type.shape
        # 检查是否有维度信息
        if not shape.dim:
            continue

        # 假设第一维 (index 0) 是 Batch Size
        dim_0 = shape.dim[0]
        
        # 移除动态参数名称 (如 'batch_size' 或 'None')
        if dim_0.HasField("dim_param"):
            dim_0.ClearField("dim_param")
        
        # 设置固定的整数值
        dim_0.dim_value = batch_size
        
        # 打印当前维度信息用于调试
        current_dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in shape.dim]
        print(f"输入 '{input_tensor.name}' 的维度已修改为: {current_dims}")

    # 3. 修改输出维度 (通常输出也带有动态 Batch，建议同步修改)
    for output_tensor in graph.output:
        shape = output_tensor.type.tensor_type.shape
        if not shape.dim:
            continue

        dim_0 = shape.dim[0]
        
        if dim_0.HasField("dim_param"):
            dim_0.ClearField("dim_param")
            
        dim_0.dim_value = batch_size
        
        current_dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in shape.dim]
        print(f"输出 '{output_tensor.name}' 的维度已修改为: {current_dims}")

    # 4. 执行 Shape Inference
    # 这一步非常关键：修改输入输出后，需要重新推断模型内部所有节点的形状
    # 否则某些推理引擎可能会因为中间层形状未知或不匹配而报错
    try:
        print("正在执行 Shape Inference...")
        model = shape_inference.infer_shapes(model)
        print("Shape Inference 完成。")
    except Exception as e:
        print(f"警告: Shape Inference 失败 (但这不一定代表模型不可用): {e}")

    # 5. 保存修改后的模型
    onnx.save(model, output_model_path)
    print(f"成功保存固定维度的模型至: {output_model_path}")

# 使用示例
if __name__ == "__main__":
    # 请替换为你自己的文件名
    input_onnx = "policy_pose0111.onnx"
    output_onnx = "policy_pose01111x285.onnx"
    
    try:
        fix_onnx_dynamic_batch(input_onnx, output_onnx, batch_size=1)
    except Exception as e:
        print(f"处理失败: {e}")
