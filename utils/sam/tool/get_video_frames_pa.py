import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def prepare_directory(path):
    """确保目录可写"""
    if os.path.exists(path):
        test_file = os.path.join(path, '.__test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except:
            raise PermissionError(f"无法写入目录: {path}")
    else:
        os.makedirs(path, exist_ok=True)


def get_valid_frame_count(directory):
    """统计目录下符合命名规则的PNG文件数量"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) 
              if f.lower().endswith('.png') and f.split('.')[0].isdigit()])

def process_video(video_path, output_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_dir = os.path.join(output_folder, video_name)
    
    # 第一级检查：目标文件夹是否存在
    if not os.path.exists(target_dir):
        return video_name, False
    
    video_subdir = os.path.join(target_dir, f"{video_name}_video")
    masks_dir = os.path.join(target_dir, f"{video_name}_masks")
    
    # 第二级检查：视频帧文件夹是否完整
    if os.path.exists(video_subdir):
        # 检查masks目录结构
        if os.path.exists(masks_dir):
            object_dirs = [d for d in os.listdir(masks_dir) 
                         if os.path.isdir(os.path.join(masks_dir, d))]
            
            if object_dirs:
                # 获取第一个对象目录
                first_object_dir = os.path.join(masks_dir, sorted(object_dirs)[0])
                
                # 比较文件数量
                video_count = get_valid_frame_count(video_subdir)
                mask_count = get_valid_frame_count(first_object_dir)
                
                if video_count == mask_count and mask_count > 0:
                    return video_name, True  # 完整处理，跳过
    
    # 需要处理的情况：清理旧数据
    if os.path.exists(video_subdir):
        shutil.rmtree(video_subdir)
    prepare_directory(video_subdir)
    
    # FFmpeg处理命令（与之前相同）
    output_pattern = os.path.join(video_subdir, "%05d.png")
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-threads', '4',
        '-i', video_path,
        '-q:v', '2',
        output_pattern
    ]
    
    try:
        subprocess.run(command, check=True, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return video_name, True
    except subprocess.CalledProcessError as e:
        # 处理失败时清理残留文件
        if os.path.exists(video_subdir):
            shutil.rmtree(video_subdir)
        print(f"\n[错误] 处理失败: {video_name}")
        print(f"错误信息: {e.stderr.decode()}")
        return video_name, False

# 主函数保持不变（与之前版本相同）
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