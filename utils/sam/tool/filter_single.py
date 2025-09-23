import json
import re

def extract_video_name(path):
    """从路径中提取视频名称"""
    match = re.search(r'(\d+-hd_\d+_\d+_\d+fps)', path)
    if not match:
        match = re.search(r'(\d+-uhd_\d+_\d+_\d+fps)', path)
    return match.group(1) if match else None

def main():
    # 读取第一个JSON（多人检测结果）
    with open('/mnt/bn/lyl/wwt/counted_data.json', 'r') as f:
        multi_data = json.load(f)
    multi_videos = set(multi_data.keys())

    # 读取第二个JSON（跟踪结果）
    with open('/mnt/bn/lyl/wwt/qualified_data.json', 'r') as f:
        tracking_data = json.load(f)

    # 初始化统计变量
    matched_count = 0
    single_qualified = []
    
    # 开始筛选
    for item in tracking_data:
        path = item['path']
        video_name = extract_video_name(path) + ".mp4"
        
        if not video_name:
            print(f"警告：无法解析路径 {path} 中的视频名称")
            continue
            
        if video_name in multi_videos:
            matched_count += 1
        else:
            single_qualified.append(item)

    # 保存结果
    with open('/mnt/bn/lyl/wwt/single_qualified_data.json', 'w') as f:
        json.dump(single_qualified, f, indent=4, ensure_ascii=False)

    # 输出统计信息
    print(f"多人视频匹配数量：{matched_count}")
    print(f"被筛除的单人视频数量：{len(single_qualified)}")
    print(f"总处理条目数：{len(tracking_data)}")
    print(f"筛除比例：{len(single_qualified)/len(tracking_data)*100:.1f}%")

if __name__ == '__main__':
    main()