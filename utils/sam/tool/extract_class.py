import os
import json

def extract_unique_classes(input_json_path, output_json_path):
    # 读取输入JSON文件
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 统计类名出现次数
    class_count = {}
    for item in data:
        path = item['path']
        # 规范化路径并获取最后一级名称
        normalized_path = os.path.normpath(path)
        class_name = os.path.basename(normalized_path).lower()
        
        # 更新统计字典
        if class_name in class_count:
            class_count[class_name] += 1
        else:
            class_count[class_name] = 1
    
    # 按出现次数排序（降序）
    sorted_class_count = dict(sorted(class_count.items(), key=lambda x: (-x[1], x[0])))
    
    # 保存到输出JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(sorted_class_count, f, indent=2, ensure_ascii=False)

import os
import json
import random

def filter_data_by_classes(input_json_path, output_json_path, class_dict):
    # 读取原始数据
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 初始化存储结构
    category_map = {cls: [] for cls in class_dict}
    
    # 遍历数据并分类
    for item in data:
        path = item['path']
        class_name = os.path.basename(os.path.normpath(path)).lower()
        
        if class_name in category_map:
            category_map[class_name].append(item)
    
    # 按需求随机筛选
    filtered_data = []
    for cls, count in class_dict.items():
        available = category_map[cls]
        if len(available) < count:
            print(f"警告: 类别 '{cls}' 只有 {len(available)} 个样本，不足要求的 {count} 个，已全部选取")
        selected = random.sample(available, min(count, len(available)))
        filtered_data.extend(selected)
    
    # 打乱顺序并保存
    random.shuffle(filtered_data)
    
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

# 使用示例
input_file = 'original_data.json'  # 替换为您的原始数据路径
output_file = 'filtered_1500.json'

class_requirements = {
    "person": 150,
    "table": 100,
    "plant": 50,
    "chair": 100,
    "laptop": 100,
    "window": 50,
    "people": 50,
    "woman": 50,
    "yoga mat": 20,
    "sofa": 100,
    "lamp": 40,
    "trees": 40,
    "plants": 50,
    "desk": 50,
    "couch": 50,
    "notebook": 50,
    "shelves": 50,
    "guitar": 50,
    "wall": 50,
    "sky": 50,
    "dress": 50,
    "shirt": 25,
    "curtain": 25,
    "dog": 50,
    "building": 50,
    "hat": 50
}

# 使用示例
input_file = '/mnt/bn/lyl/wwt/qualified_data.json'  # 替换为您的输入文件路径
output_file = '/mnt/bn/lyl/wwt/filterd_1500.json'
filter_data_by_classes(input_file, output_file, class_requirements)