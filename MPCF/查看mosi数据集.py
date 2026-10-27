import pickle
import os

# 使用相对路径，更加灵活
dataset_path = r"d:\PythonWorkplace\MPCF-main\MPCF\datasets\MOSI"  # 相对于当前MPCF目录
file_name = "dev.pkl"  # 可以替换为train.pkl或test.pkl

# 加载数据
with open(os.path.join(dataset_path, file_name), "rb") as f:
    # 解决pickle编码问题（旧版本可能需要latin1编码）
    data = pickle.load(f, encoding="latin1")

# 查看数据整体结构
print(f"数据集共包含 {len(data)} 个样本")
print(f"第223个样本的类型: {type(data[222])}")
print(f"第223个样本包含 {len(data[222])} 个元素\n")

# 元组格式的MOSI样本通常按以下顺序存储（不同预处理版本可能有差异）
# 这里提供两种最常见的解析方案，根据实际情况选择

# 方案1：常见的5元素元组格式（文本特征, 音频特征, 视觉特征, 标签, 元数据）
try:
    sample = data[222]  # 修改为第223个样本
    text_feat, audio_feat, visual_feat, label, meta = sample
    
    # 解析元数据（通常包含文本token、长度等信息）
    text_tokens = meta.get('tokens', []) if isinstance(meta, dict) else []
    text_len = len(text_tokens) if text_tokens else 0
    audio_len = audio_feat.shape[0] if audio_feat is not None else 0
    visual_len = visual_feat.shape[0] if visual_feat is not None else 0

    print("方案1解析成功（5元素元组）：")
    print(f"文本特征形状: {text_feat.shape}")
    print(f"音频特征形状: {audio_feat.shape}")
    print(f"视觉特征形状: {visual_feat.shape}")
    print(f"情感标签: {label}")
    print(f"文本token示例: {text_tokens[:5]}")  # 打印前5个token

    # 组装input_data
    input_data = {
        "text": {
            "tokens": text_tokens,
            "features": text_feat,
            "length": text_len
        },
        "audio": {
            "features": audio_feat,
            "length": audio_len
        },
        "visual": {
            "features": visual_feat,
            "length": visual_len
        },
        "label": label
    }

except (ValueError, IndexError):
    # 方案2：4元素元组格式（文本特征, 音频特征, 视觉特征, 标签）
    try:
        sample = data[222]  # 修改为第223个样本
        if len(sample) == 3:
            # 3元素格式：可能是 (text_tokens, text_feat, audio_feat) 或其他组合
            element1, element2, element3 = sample
            print("3元素格式解析：")
            print(f"元素1类型: {type(element1)}, 内容: {element1 if not hasattr(element1, 'shape') else f'数组形状: {element1.shape}'}")
            print(f"元素2类型: {type(element2)}, 形状: {element2.shape if hasattr(element2, 'shape') else '非数组'}")
            print(f"元素3类型: {type(element3)}, 形状: {element3.shape if hasattr(element3, 'shape') else '非数组'}")
        text_feat, audio_feat, visual_feat, label = sample
        
        # 估算长度（如果元数据中没有则用特征形状推断）
        text_len = text_feat.shape[0] if text_feat.ndim > 0 else 0
        audio_len = audio_feat.shape[0] if audio_feat.ndim > 0 else 0
        visual_len = visual_feat.shape[0] if visual_feat.ndim > 0 else 0

        print("方案2解析成功（4元素元组）：")
        print(f"文本特征形状: {text_feat.shape}")
        print(f"音频特征形状: {audio_feat.shape}")
        print(f"视觉特征形状: {visual_feat.shape}")
        print(f"情感标签: {label}")

        # 组装input_data
        input_data = {
            "text": {
                "features": text_feat,
                "length": text_len
            },
            "audio": {
                "features": audio_feat,
                "length": audio_len
            },
            "visual": {
                "features": visual_feat,
                "length": visual_len
            },
            "label": label
        }

    except Exception as e:
        print(f"解析失败: {str(e)}")
        print("请手动查看样本结构：")
        print("第一个样本的前3个元素：", sample[:3])
