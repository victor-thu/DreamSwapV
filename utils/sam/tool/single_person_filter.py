import json

# 1. 读取原始 JSON 数据
input_file = '/mnt/bn/lyl/wwt/single_qualified_data.json'  # 原始 JSON 文件名
output_file = '/mnt/bn/lyl/wwt/single_person_data.json'  # 输出文件名

# 定义与人物相关的类目
person_related_classes = ["person", "Person", "Woman", "Man", "woman", "man", "people"]

# 读取 JSON 文件
with open(input_file, 'r') as f:
    data = json.load(f)

# 2. 筛选与人物相关的条目
filtered_data = []
for item in data:
    # 从路径中提取类目名
    path_parts = item['path'].split('/')
    class_name = path_parts[-1]  # 类目名称通常位于路径的最后一部分

    # 检查类目名称是否在人物相关类目列表中
    if any(person_class.lower() in class_name.lower() for person_class in person_related_classes):
        filtered_data.append(item)

# 3. 将筛选后的数据写入新的 JSON 文件
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"筛选后的数据已保存到 {output_file}")
