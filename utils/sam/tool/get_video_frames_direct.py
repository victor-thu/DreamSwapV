import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_frame_count(folder):
    """获取指定目录下.png或.jpg文件的数量"""
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) 
        if f.lower().endswith(('.png', '.jpg'))])

def get_reference_folder(video_name):
    """构建参考路径并获取其第一个子文件夹的文件数"""
    base_path = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results"
    ref_dir = os.path.join(base_path, f"{video_name}", f"{video_name}_masks")
    
    if not os.path.exists(ref_dir):
        return 0
    
    # 获取第一个子文件夹
    subfolders = [d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d))]
    if not subfolders:
        return 0
    
    first_subfolder = os.path.join(ref_dir, subfolders[0])
    return get_frame_count(first_subfolder)

def extract_frames(video_path, output_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_folder, video_name, f"{video_name}_video")
    
    # 新增复杂检查逻辑
    try:
        # 检查帧目录是否存在
        if os.path.exists(frame_dir):
            current_count = get_frame_count(frame_dir)
            reference_count = get_reference_folder(video_name)
            
            # 检查是否需要重新处理
            if current_count == reference_count and reference_count > 0:
                return video_name, "skipped"
    except Exception as e:
        print(f"[检查异常] {video_name}: {str(e)}")
        return video_name, "failed"

    # 创建目录并处理
    os.makedirs(frame_dir, exist_ok=True)
    
    # FFmpeg命令
    command = [
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'info',
        '-threads', '4',
        '-i', video_path,
        '-q:v', '2',
        '-pix_fmt', 'yuvj420p',
        os.path.join(frame_dir, '%05d.jpg')
    ]
    
    try:
        subprocess.run(command, check=True, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return video_name, "success"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode().split('\n')[0]
        print(f"[失败] {video_name}: {error_msg}")
        return video_name, "failed"

def main(video_folder, output_folder):
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                  if f.lower().endswith('.mp4')]
    total = len(video_files)
    stats = {
        "total": len(video_files),
        "success": 0,
        "failed": [],
        "skipped": 0
    }
    
    print(f"开始处理 {stats['total']} 个视频...\n")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_frames, vf, output_folder): vf 
                  for vf in video_files}
        
        # 初始化进度显示
        processed = 0
        
        for future in as_completed(futures):
            video_name, status = future.result()
            processed += 1  # 计数器
            
            # 进度显示
            print(f"\n进度：{processed}/{total} [", end="", flush=True)
            
            # 处理状态
            if status == "success":
                stats["success"] += 1
                print("✓", end="")
            elif status == "failed":
                stats["failed"].append(video_name)
                print("✗", end="")
            elif status == "skipped":
                stats["skipped"] += 1
                print("→", end="")
            
            print(f"] {video_name}", flush=True)

    print("\n" + "="*50)
    print(f"处理统计：")
    print(f"总任务数: {stats['total']}")
    print(f"成功处理: {stats['success']}")
    print(f"跳过处理: {stats['skipped']}")
    print(f"处理失败: {len(stats['failed'])}")
    
    if stats['failed']:
        print("\n失败视频列表:")
        print("\n".join(stats['failed']))

if __name__ == "__main__":
    VIDEO_FOLDER = "/mnt/bn/lyl/wwt/humanvid"    # 修改为你的视频文件夹路径
    OUTPUT_FOLDER = "/mnt/bn/aigc-algorithm-group/weitao/humanvid_data" # 修改为输出文件夹路径

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    main(VIDEO_FOLDER, OUTPUT_FOLDER)