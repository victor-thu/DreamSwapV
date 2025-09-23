import cv2
import numpy as np
import os

def calculate_motion_metric(mask_frames, threshold=0.1):
    """
    计算mask帧中物体的运动指标，并判断是否达到阈值。
    
    参数:
        mask_frames (list): 包含mask帧的列表，每个mask帧为二值图像（黑白图像）。
        threshold (float): 运动指标的阈值，默认为0.1（即10%的变化）。
        
    返回:
        dict: 包含总运动指标和是否达标的字典。
    """
    if len(mask_frames) < 2:
        raise ValueError("至少需要两帧mask才能计算运动指标。")
    
    # 初始化变量
    total_pixel_change = 0
    total_pixels = 0
    
    # 遍历相邻帧对
    for i in range(len(mask_frames) - 1):
        prev_frame = mask_frames[i]
        curr_frame = mask_frames[i + 1]
        
        # 确保两帧大小一致
        if prev_frame.shape != curr_frame.shape:
            raise ValueError("所有mask帧的尺寸必须相同。")
        
        # 计算像素变化（异或操作）
        pixel_change = cv2.bitwise_xor(prev_frame, curr_frame)
        
        # 统计变化的像素数
        changed_pixels = np.sum(pixel_change > 0)
        total_pixel_change += changed_pixels
        
        # 统计总像素数（以第一帧为准）
        if i == 0:
            total_pixels = prev_frame.size
    
    # 计算运动指标（变化的像素占总像素的比例）
    motion_metric = total_pixel_change / total_pixels
    
    # 判断是否达标
    is_sufficient_motion = motion_metric >= threshold
    
    # 返回结果
    return {
        "motion_metric": motion_metric,
        "is_sufficient_motion": is_sufficient_motion
    }

def load_mask_frames_from_directory(directory_path):
    """
    从指定目录加载mask帧。
    
    参数:
        directory_path (str): 包含mask帧的目录路径。
        
    返回:
        list: 包含mask帧的列表。
    """
    mask_frames = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(directory_path, filename)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取文件: {file_path}")
            mask_frames.append(mask)
    return mask_frames

# 示例用法
if __name__ == "__main__":
    # 设置参数
    mask_directory = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results/1122526-hd_1920_1080_25fps/1122526-hd_1920_1080_25fps_masks/napkin holder"  # 替换为你的mask帧目录路径
    threshold = 0.1  # 设定阈值，表示10%的像素变化
    
    # 加载mask帧
    mask_frames = load_mask_frames_from_directory(mask_directory)
    
    # 计算运动指标
    result = calculate_motion_metric(mask_frames, threshold)
    
    # 输出结果
    print(f"运动指标: {result['motion_metric']:.4f}")
    print(f"是否达到阈值: {'是' if result['is_sufficient_motion'] else '否'}")