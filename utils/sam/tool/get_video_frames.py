import os
import cv2

def extract_frames(video_folder, result_folder):
    # 获取所有mp4文件
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    skipped = 0

    for video_file in video_files:
        # 提取基础文件名（不含扩展）
        base_name = os.path.splitext(video_file)[0]
        target_dir = os.path.join(result_folder, base_name)
        
        # 检查目标文件夹是否存在
        if not os.path.exists(target_dir):
            print(f"文件夹 {base_name} 不存在，跳过该视频")
            skipped += 1
            continue
        
        # 创建视频帧保存目录
        video_subdir = os.path.join(target_dir, f"{base_name}_video")
        os.makedirs(video_subdir, exist_ok=True)
        
        # 处理视频
        video_path = os.path.join(video_folder, video_file)
        vidcap = cv2.VideoCapture(video_path)
        
        if not vidcap.isOpened():
            print(f"无法打开视频文件 {video_file}，跳过处理")
            vidcap.release()
            continue
        
        success, image = vidcap.read()
        count = 0
        
        while success:
            frame_name = os.path.join(video_subdir, f"{count:05d}.png")
            cv2.imwrite(frame_name, image)
            success, image = vidcap.read()
            count += 1
        
        vidcap.release()
        print(f"处理完成：{video_file}，共提取 {count} 帧")

    print(f"\n处理完成！跳过 {skipped} 个视频")

if __name__ == "__main__":
    # 设置路径（请根据实际情况修改）
    video_folder = "/mnt/bn/lyl/wwt/humanvid"
    result_folder = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results"

    extract_frames(video_folder, result_folder)