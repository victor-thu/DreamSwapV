import json
from transformers import pipeline
from collections import defaultdict
from accelerate import dispatch_model, infer_auto_device_map
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 检查是否有可用的GPU
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available. Please ensure CUDA is installed and GPUs are accessible.")
else:
    print("CUDA is available. Proceeding with GPU support.")

# 获取可用的GPU数量
num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

# 加载模型和分词器
model_name = "facebook/bart-large-mnli"
print(f"Loading model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

# 自动分配模型到多个GPU
print("Inferring device map for multi-GPU allocation...")
device_map = infer_auto_device_map(model)
print(f"Device map inferred: {device_map}")
print("Dispatching model to devices...")
model = dispatch_model(model, device_map=device_map)
print("Model dispatched to devices successfully.")

# 创建分类器，并指定多GPU支持
print("Creating zero-shot classification pipeline...")
classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device_map=device_map  # 使用多GPU分配
)
print("Pipeline created successfully.")

def categorize_object(object_name):
    """使用预训练模型进行语义分类"""
    categories = ["person", "clothing", "small object", "other"]
    
    print(f"Classifying object: {object_name}")
    results = classifier(
        object_name,
        candidate_labels=categories,
        hypothesis_template="This object belongs to the category of {}."
    )
    
    # 返回最可能的类别（去除括号内的说明）
    category = results['labels'][0]
    print(f"Object '{object_name}' classified as: {category}")
    return category

# 读取原始数据
data_path = '/mnt/bn/lyl/wwt/qualified_data.json'
print(f"Reading input data from: {data_path}")
with open(data_path, 'r') as f:
    data = json.load(f)
print(f"Successfully loaded {len(data)} items from the input file.")

# 初始化分类存储和统计字典
classified_data = defaultdict(list)
stats = defaultdict(int)

# 分类并整理数据
print("Starting classification process...")
for idx, item in enumerate(data):
    path = item['path']
    object_name = path.split('/')[-1].lower()
    print(f"[Item {idx + 1}/{len(data)}] Processing object: {object_name}")
    
    # 分类当前物体
    category = categorize_object(object_name)
    
    # 添加到分类结果
    classified_data[category].append({
        "path": item['path'],
        "frame_count": item['frame_count']
    })
    
    # 更新统计
    stats[category] += 1

print("Classification process completed.")

# 保存分类结果到新JSON文件
output_path = '/mnt/bn/lyl/wwt/qualified_classified_results.json'
print(f"Saving classification results to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(classified_data, f, ensure_ascii=False, indent=2)
print("Classification results saved successfully.")

# 输出统计结果
print("分类统计结果：")
for category, count in stats.items():
    print(f"{category}: {count}")