import os
import shutil

def delete_subsub_video_folders(root_dir):
    to_delete = []
    
    # 遍历根目录下的所有一级子目录
    for dir1 in os.listdir(root_dir):
        dir1_path = os.path.join(root_dir, dir1)
        
        # 确保是一级子目录
        if not os.path.isdir(dir1_path):
            continue
            
        # 遍历一级子目录下的二级子目录（子子文件夹）
        for dir2 in os.listdir(dir1_path):
            dir2_path = os.path.join(dir1_path, dir2)
            
            # 检查是否为二级子目录且以小写video结尾
            if os.path.isdir(dir2_path) and dir2.endswith('video'):
                to_delete.append(dir2_path)
    
    # 安全确认
    if not to_delete:
        print("未找到符合条件的子子文件夹")
        return

    print(f"即将删除 {len(to_delete)} 个子子文件夹：")
    for path in to_delete:
        print(f"  - {path}")
    
    confirm = input("\n确认删除？(y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return

    # 执行删除
    for path in to_delete:
        try:
            shutil.rmtree(path)
            print(f"已删除: {path}")
        except Exception as e:
            print(f"删除失败: {path} ({str(e)})")

if __name__ == "__main__":
    # 替换为你的根目录路径
    root_directory = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results"
    delete_subsub_video_folders(root_directory)