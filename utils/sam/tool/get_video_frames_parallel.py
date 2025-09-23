import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_video(video_path, output_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_dir = os.path.join(output_folder, video_name)
    
    # 检查目标文件夹是否存在
    if not os.path.exists(target_dir):
        return video_name, False  # 返回需要跳过的视频名和状态
    
    # 创建视频帧存储目录
    video_subdir = os.path.join(target_dir, f"{video_name}_video")
    os.makedirs(video_subdir, exist_ok=True)
    
    # 构建FFmpeg命令
    output_pattern = os.path.join(video_subdir, "%05d.png")
    command = [
        'ffmpeg',
        '-hide_banner',          # 隐藏FFmpeg横幅
        '-loglevel', 'error',    # 只显示错误信息
        '-hwaccel', 'cuda',      # 启用GPU加速
        '-i', video_path, 
        '-q:v', '2', 
        output_pattern
    ]
    
    try:
        # 执行FFmpeg命令
        subprocess.run(command, check=True, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return video_name, True   # 返回成功状态
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 处理失败: {video_name}")
        print(f"错误信息: {e.stderr.decode()}")
        return video_name, False

def main(video_folder, output_folder):
    # 收集所有视频文件
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) 
                  if f.lower().endswith('.mp4')]
    total_videos = len(video_files)
    skipped = []
    processed = 0

    print(f"开始处理 {total_videos} 个视频...\n")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_video, vf, output_folder): vf for vf in video_files}
        
        for future in as_completed(futures):
            video_name, success = future.result()
            processed += 1
            
            # 实时进度显示
            progress = f"[{processed}/{total_videos}] "
            
            if success:
                print(f"{progress}✓ 处理完成: {video_name}")
            else:
                print(f"{progress}✗ 跳过: {video_name}")
                skipped.append(video_name)

    # 最终统计报告
    print("\n" + "="*50)
    print(f"处理完成！共处理 {total_videos} 个视频")
    print(f"成功: {total_videos - len(skipped)} 个")
    print(f"跳过: {len(skipped)} 个")
    
    if skipped:
        print("\n跳过的视频列表:")
        print("\n".join(skipped))

if __name__ == "__main__":
    VIDEO_FOLDER = "/mnt/bn/lyl/wwt/humanvid"    # 修改为你的视频文件夹路径
    OUTPUT_FOLDER = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results" # 修改为输出文件夹路径

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    main(VIDEO_FOLDER, OUTPUT_FOLDER)