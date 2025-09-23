import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 配置参数
MULTI_JSON = '/mnt/bn/lyl/wwt/counted_data.json'
OUTPUT_JSON = '/mnt/bn/lyl/wwt/unqualified_data.json'
POSE_ROOT = '/mnt/bn/aigc-algorithm-group/weitao/humanvid_data'
FRAME_SKIP = 5       # 帧采样间隔
PIXEL_THRESHOLD = 30  # 像素位移阈值（根据分辨率调整）
MIN_JUMP_FRAMES = 2   # 最小连续跳变帧数

def get_skeleton_mask(frame):
    """提取骨骼区域掩码"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return mask

def calculate_centroid(mask):
    """计算骨骼区域的质心"""
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return (0, 0)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def check_pose_jump(video_name):
    """基于像素位移的跳变检测"""
    video_name = video_name.split('.')[0]
    pose_dir = os.path.join(POSE_ROOT, f"{video_name}",
                            f"{video_name}_video_pose")
    
    if not os.path.exists(pose_dir):
        return None

    frame_files = sorted(Path(pose_dir).glob('*.png')) + sorted(Path(pose_dir).glob('*.jpg'))
    if len(frame_files) < 2:
        return None

    # 预加载参考帧
    prev_frame = cv2.imread(str(frame_files[0]))
    prev_mask = get_skeleton_mask(prev_frame)
    prev_centroid = calculate_centroid(prev_mask)
    
    jump_count = 0

    for i in range(FRAME_SKIP, len(frame_files), FRAME_SKIP):
        curr_frame = cv2.imread(str(frame_files[i]))
        if curr_frame is None:
            return None
        curr_mask = get_skeleton_mask(curr_frame)
        curr_centroid = calculate_centroid(curr_mask)
        
        # 计算质心位移
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        displacement = np.sqrt(dx**2 + dy**2)
        
        if displacement > PIXEL_THRESHOLD:
            jump_count += 1
            if jump_count >= MIN_JUMP_FRAMES:
                print(f"视频 {video_name} 检测到跳变！")
                return True
        else:
            jump_count = 0
        
        # 更新参考帧
        prev_mask = curr_mask
        prev_centroid = curr_centroid

        # 提前终止条件
        if i > len(frame_files)//2 and jump_count == 0:
            break

    return False

def main():
    with open(MULTI_JSON, 'r') as f:
        multi_data = json.load(f)
    
    unqualified = []
    total_videos = len(multi_data)
    
    with tqdm(total=total_videos, desc="检测进度") as pbar:
        for video_name, count in multi_data.items():
            has_jump = check_pose_jump(video_name)
            if has_jump:
                unqualified.append({
                    "video_name": video_name,
                    "people_count": count
                })
            pbar.update(1)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(unqualified, f, indent=4, ensure_ascii=False)
    
    print(f"检测完成！不合格视频数量：{len(unqualified)}/{total_videos}")

if __name__ == '__main__':
    main()