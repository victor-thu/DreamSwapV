import os

def is_empty_folder(folder_path):
    """
    检查一个文件夹是否为空（包括其所有子文件夹）。
    :param folder_path: 文件夹路径
    :return: True 如果文件夹为空，否则 False
    """
    for root, dirs, files in os.walk(folder_path):
        if files:  # 如果当前文件夹中有文件，则不是空文件夹
            return False
    return True

def find_top_level_empty_folders(base_dir):
    """
    查找指定目录下的一级空文件夹。
    :param base_dir: 要检查的根目录
    :return: 空文件夹路径列表
    """
    empty_folders = []
    
    # 列出根目录下的所有一级子文件夹
    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{base_dir}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for directory '{base_dir}'.")
        return []
    
    for entry in entries:
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path):  # 只处理文件夹
            if is_empty_folder(entry_path):
                empty_folders.append(entry_path)
    
    return empty_folders

if __name__ == "__main__":
    # 设置要检查的根目录
    base_directory = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results"  # 替换为你的文件夹路径
    
    # 查找一级空文件夹
    empty_folders = find_top_level_empty_folders(base_directory)
    
    if empty_folders:
        print("Top-level empty folders found:")
        for folder in empty_folders:
            print(folder)
    else:
        print("No top-level empty folders found.")