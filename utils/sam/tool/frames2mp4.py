import os
import subprocess
import cv2
import numpy as np

def find_png_pattern(input_folder):
    """
    自动检测 PNG 文件的命名模式。
    """
    files = sorted(os.listdir(input_folder))
    png_files = [f for f in files if f.endswith(".png")]
    
    if not png_files:
        raise ValueError(f"文件夹中未找到 PNG 文件: {input_folder}")
    
    # 提取文件名模式
    base_name = os.path.splitext(png_files[0])[0]
    digits = ''.join(filter(str.isdigit, base_name))
    prefix = base_name[:base_name.index(digits)] if digits else ''
    
    # 检查是否有前导零
    has_leading_zeros = all(len(f) == len(png_files[0]) for f in png_files)
    
    # 构造通配符模式
    if has_leading_zeros:
        pattern = f"{prefix}%0{len(digits)}d.png"
    else:
        pattern = f"{prefix}%d.png"
    
    return os.path.join(input_folder, pattern)

def process_mask_to_bounding_box(mask):
    """
    将 mask 的白色区域扩展到其 bounding box 的形状，保持原始 mask 的尺寸不变。

    参数:
        mask (numpy.ndarray): 输入的 mask 图像（灰度图）。

    返回:
        numpy.ndarray: 处理后的 mask 图像。
    """
    # 检查是否为全黑图像
    if not np.any(mask):
        print("警告: 输入的 mask 是全黑图像，未进行边界框扩展。")
        return mask
    
    # 计算非零像素的边界框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    print(xmin, ymin, xmax, ymax)
    
    # 创建一个与原始 mask 形状相同的空白图像
    extended_mask = np.zeros_like(mask)
    
    # 将裁剪后的白色区域扩展到边界框大小，并放回原始位置
    extended_mask[ymin:ymax+1, xmin:xmax+1] = 255  # 白色区域填充为 255
    
    return extended_mask

def convert_sequence_to_mp4(
    input_video_folder, 
    input_mask_folder, 
    output_folder, 
    save_mask_video=False,  # 新增参数：是否保存mask视频
    fps=30, 
    frame_length=81
):
    """
    将原始视频帧和处理后的 mask 结合，生成最终视频，并可选保存增强后的mask视频

    新增参数:
        save_mask_video (bool): 是否保存增强后的mask视频
    """
    if not os.path.isdir(input_video_folder):
        raise ValueError(f"输入视频文件夹不存在: {input_video_folder}")
    if not os.path.isdir(input_mask_folder):
        raise ValueError(f"输入mask文件夹不存在: {input_mask_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 视频命名
    video_name = f"processed_{os.path.basename(input_video_folder)}.mp4"
    output_video_file = os.path.join(output_folder, video_name)
    
    mask_video_name = f"mask_{os.path.basename(input_video_folder)}.mp4"
    output_mask_video_file = os.path.join(output_folder, mask_video_name)
    
    # 获取并排序文件列表
    video_files = sorted([f for f in os.listdir(input_video_folder) if f.endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(input_mask_folder) if f.endswith(".png")])
    
    # 检查文件数量是否一致
    if len(video_files) != len(mask_files):
        raise ValueError(f"视频帧({len(video_files)})和mask帧({len(mask_files)})数量不一致")
    
    processed_frames = []
    processed_masks = []  # 用于保存增强后的mask
    
    for i in range(min(len(video_files), frame_length)):
        video_path = os.path.join(input_video_folder, video_files[i])
        mask_path = os.path.join(input_mask_folder, mask_files[i])
        
        # 读取原始视频帧和mask
        video_frame = cv2.imread(video_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if video_frame is None:
            raise ValueError(f"无法读取视频帧: {video_path}")
        if mask is None:
            raise ValueError(f"无法读取mask: {mask_path}")
        
        # 处理mask
        # processed_mask = process_mask_to_bounding_box(mask)
        processed_mask = mask
        
        # 应用mask到视频帧
        processed_frame = video_frame.copy()
        processed_frame[processed_mask == 255] = [128, 128, 128]  # 灰色
        
        processed_frames.append(processed_frame)
        if save_mask_video:
            processed_masks.append(processed_mask)
    
    if not processed_frames:
        raise ValueError("没有可用的帧！")
    
    # 生成主视频
    height, width, _ = processed_frames[0].shape
    process_video = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-frames:v", str(frame_length),
            output_video_file
        ],
        stdin=subprocess.PIPE
    )
    
    # 生成mask视频（如果需要）
    process_mask = None
    if save_mask_video:
        process_mask = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "gray",  # 使用灰度格式
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-frames:v", str(frame_length),
                output_mask_video_file
            ],
            stdin=subprocess.PIPE
        )
    
    try:
        # 同时写入两个视频流
        for i, frame in enumerate(processed_frames):
            # 写入主视频
            process_video.stdin.write(frame.tobytes())
            
            # 写入mask视频
            if save_mask_video and i < len(processed_masks):
                mask_frame = processed_masks[i]
                process_mask.stdin.write(mask_frame.tobytes())
    except Exception as e:
        print(f"写入帧时出错: {e}")
    finally:
        process_video.stdin.close()
        if save_mask_video:
            process_mask.stdin.close()
        process_video.wait()
        if save_mask_video:
            process_mask.wait()
    
    print(f"主视频已生成: {output_video_file}")
    if save_mask_video:
        print(f"Mask视频已生成: {output_mask_video_file}")

# 示例用法
if __name__ == "__main__":
    input_video_folder = "/mnt/bn/aigc-algorithm-group/weitao/humanvid_data/5837589-hd_1920_1080_24fps/5837589-hd_1920_1080_24fps_video"  # 替换为实际视频帧路径
    input_mask_folder = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results/5837589-hd_1920_1080_24fps/5837589-hd_1920_1080_24fps_masks/Person"
    output_folder = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/mask_videos"
    fps = 24
    frame_length = 81

    convert_sequence_to_mp4(input_video_folder, input_mask_folder, output_folder, True, fps, frame_length)