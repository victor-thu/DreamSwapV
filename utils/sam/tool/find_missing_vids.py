import os
import json

# 读取原始JSON文件中的视频路径
with open('/mnt/bn/lyl/wwt/unique_video_paths.json', 'r') as f:
    video_paths = json.load(f)

# 提取所有vid到集合中
existing_vids = set()
for path in video_paths:
    vid = os.path.basename(os.path.dirname(path))
    existing_vids.add(vid)

# 指定要检查的目标文件夹路径（请根据实际情况修改）
target_dir = '/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results'

# 收集不存在于原始JSON中的子文件夹路径
missing_vid_paths = []
for entry in os.scandir(target_dir):
    if entry.is_dir():
        vid = entry.name
        if vid not in existing_vids:
            missing_vid_paths.append(entry.path)

# 保存结果到新JSON文件
with open('/mnt/bn/lyl/wwt/missing_video_paths.json', 'w') as f:
    json.dump(missing_vid_paths, f, indent=2)

print(f"发现 {len(missing_vid_paths)} 个缺失的vid文件夹，已保存到 missing_vids.json")