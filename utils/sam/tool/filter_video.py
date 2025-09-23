import json
from pathlib import Path

def extract_video_name(path_str):
    """从路径中提取视频名称"""
    # 将路径转换为Path对象
    path = Path(path_str)
    
    # 获取tracking_results目录下的视频目录名
    # 路径结构：.../tracking_results/<video_name>/<video_name>_masks/...
    parts = path.parts
    try:
        # 找到tracking_results的索引
        tracking_index = parts.index("tracking_results")
        video_name = parts[tracking_index + 1]
        return video_name
    except ValueError:
        print(f"路径格式错误：{path_str}")
        return None

def generate_video_paths(input_json, output_json):
    # 读取输入JSON
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # 提取并去重视频名
    video_names = set()
    for item in data:
        video_name = extract_video_name(item['path'])
        if video_name:
            video_names.add(video_name)
    
    # 生成目标路径格式
    output_data = [
        f"/mnt/bn/aigc-algorithm-group/weitao/humanvid_data/{name}/{name}_video"
        for name in video_names
    ]
    
    # 保存到新JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功提取{len(output_data)}个唯一视频路径，保存至：{output_json}")

# 使用示例
generate_video_paths(
    input_json="/mnt/bn/lyl/wwt/qualified_data.json",
    output_json="/mnt/bn/lyl/wwt/unique_video_paths.json"
)