import os
import cv2
import numpy as np
import json
from datetime import datetime

def calculate_motion_metric(mask_frames, threshold=0.1):
    if len(mask_frames) < 2:
        return {"motion_metric": 0.0, "is_sufficient_motion": False}
    
    total_motion = 0.0
    valid_pairs = 0

    for i in range(len(mask_frames)-1):
        prev = mask_frames[i]
        curr = mask_frames[i+1]
        
        if prev.shape != curr.shape:
            raise ValueError("Mask尺寸不一致")
            
        # 计算当前帧对的物体面积
        prev_area = np.sum(prev == 255)
        curr_area = np.sum(curr == 255)
        avg_area = (prev_area + curr_area) / 2
        
        # 跳过无效区域（两帧面积均为0）
        if avg_area == 0:
            continue
            
        # 计算像素变化
        diff = cv2.bitwise_xor(prev, curr)
        changed_pixels = np.sum(diff > 0)
        
        # 计算相对运动比例
        motion_ratio = changed_pixels / avg_area
        total_motion += motion_ratio
        valid_pairs += 1

    # 计算平均运动指标
    motion_metric = total_motion / valid_pairs if valid_pairs else 0.0
    return {
        "motion_metric": motion_metric,
        "is_sufficient_motion": motion_metric >= threshold
    }

def main(mask_root, video_root, area_threshold, motion_threshold, output_file):
    start_time = datetime.now()
    results = []
    total_videos = 0
    total_objects = 0
    qualified_count = 0
    
    # 预统计总任务量
    video_list = []
    for video_name in os.listdir(mask_root):
        mask_dir = os.path.join(mask_root, video_name, f"{video_name}_masks")
        video_dir = os.path.join(video_root, video_name, f"{video_name}_video")
        if os.path.exists(mask_dir) and os.path.exists(video_dir):
            video_list.append(video_name)
            total_objects += len(os.listdir(mask_dir))
    total_videos = len(video_list)
    
    print(f"开始筛选任务 - 总视频数: {total_videos} | 总物体数: {total_objects}")
    
    for video_idx, video_name in enumerate(video_list, 1):
        mask_dir = os.path.join(mask_root, video_name, f"{video_name}_masks")
        video_dir = os.path.join(video_root, video_name, f"{video_name}_video")
        
        # 获取视频帧数
        video_frames = [f for f in os.listdir(video_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        video_count = len(video_frames)
        
        # 获取物体列表
        objects = [d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]
        current_video_qualified = 0
        
        print(f"\n[{datetime.now()}] 正在处理视频 {video_idx}/{total_videos}: {video_name}")
        print(f"  总物体数: {len(objects)} | 视频帧数: {video_count}")
        
        for obj_idx, obj_name in enumerate(objects, 1):
            obj_path = os.path.join(mask_dir, obj_name)
            status = "处理中"
            
            try:
                # 规则1：帧数匹配
                mask_files = sorted([f for f in os.listdir(obj_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                mask_count = len(mask_files)
                if mask_count != video_count:
                    status = "帧数不匹配"
                    print(f"\r    处理物体 {obj_idx}/{len(objects)}: {obj_name} [{status}]", end="")
                    continue
                
                # 采样配置（0,10,20,30帧）
                selected_indices = [0, 10, 20, 30]
                selected_files = [mask_files[i] for i in selected_indices if i < mask_count]
                
                # 加载采样帧
                mask_frames = []
                for filename in selected_files:
                    file_path = os.path.join(obj_path, filename)
                    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"无法读取文件: {file_path}")
                    mask_frames.append(mask)
                
                # 规则2：面积占比
                total_area = 0
                for mask in mask_frames:
                    white = np.sum(mask == 255)
                    total_area += white / mask.size
                area_ratio = total_area / len(mask_frames) if mask_frames else 0.0
                
                if area_ratio < area_threshold:
                    status = "面积不足"
                    print(f"\r    处理物体 {obj_idx}/{len(objects)}: {obj_name} [{status}]", end="")
                    continue
                
                # 规则3：运动指标
                if len(mask_frames) < 2:
                    status = "运动不足（帧数不足）"
                    print(f"\r    处理物体 {obj_idx}/{len(objects)}: {obj_name} [{status}]", end="")
                    continue
                
                motion_result = calculate_motion_metric(mask_frames, motion_threshold)
                if not motion_result["is_sufficient_motion"]:
                    status = "运动不足"
                    print(f"\r    处理物体 {obj_idx}/{len(objects)}: {obj_name} [{status}]", end="")
                    continue
                
                # 记录合格结果
                results.append({
                    "path": os.path.abspath(obj_path),
                    "frame_count": mask_count,
                })
                current_video_qualified +=1
                qualified_count +=1
                status = "合格"
                print(f"\r    处理物体 {obj_idx}/{len(objects)}: {obj_name} [{status}]", end="")
                
            except Exception as e:
                status = f"错误: {str(e)}"
                print(f"\r    处理物体 {obj_idx}/{len(objects)}: {obj_name} [{status}]", end="")
            
        print(f"\n  当前视频合格物体: {current_video_qualified}/{len(objects)}")
        
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    elapsed = datetime.now() - start_time
    print("\n" + "="*60)
    print(f"筛选完成！总耗时: {elapsed}")
    print(f"总处理视频: {total_videos} | 总处理物体: {total_objects}")
    print(f"合格物体数: {qualified_count}")
    print(f"结果已保存到: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 配置参数
    MASK_ROOT = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results"  # 替换为mask根目录
    VIDEO_ROOT = "/mnt/bn/aigc-algorithm-group/weitao/humanvid_data"  # 替换为视频帧根目录
    AREA_THRESHOLD = 0.01  # 面积阈值
    MOTION_THRESHOLD = 0.1  # 运动阈值
    OUTPUT_FILE = "/mnt/bn/lyl/wwt/qualified_data.json"  # 输出文件
    
    main(MASK_ROOT, VIDEO_ROOT, AREA_THRESHOLD, MOTION_THRESHOLD, OUTPUT_FILE)
    print(f"筛选完成，结果已保存到{OUTPUT_FILE}")