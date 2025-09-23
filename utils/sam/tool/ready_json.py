import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def remove_matches(source_list, check_list):
    # 创建检查集合（统一转换为字符串）
    check_set = set(str(item).strip() for item in check_list)
    
    # 过滤源列表（保留不在检查集合中的项）
    return [item for item in source_list 
            if str(item).strip() not in check_set]

def main():
    source_file = '/mnt/bn/lyl/wwt/unique_video_paths_2.json'
    check_file = '/mnt/bn/lyl/wwt/ready_video.json'
    output_file = '/mnt/bn/lyl/wwt/unique_video_paths_2.json'

    source_data = load_json(source_file)
    check_data = load_json(check_file)

    # 确保都是列表类型
    if not isinstance(source_data, list) or not isinstance(check_data, list):
        raise ValueError("文件内容必须为JSON数组")

    filtered = remove_matches(source_data, check_data)
    save_json(filtered, output_file)
    print(f"处理完成！共保留{len(filtered)}条，结果保存至{output_file}")

if __name__ == "__main__":
    main()