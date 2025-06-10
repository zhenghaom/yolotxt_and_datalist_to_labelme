import json
import os

def convert_points_to_two(points):
    """
    将四个点转换为两个点，表示矩形的左上角和右下角。
    """
    if len(points) > 2:
        # 找到最小的 x 和 y 值，以及最大的 x 和 y 值
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)
        # 返回左上角和右下角的点
        return [[min_x, min_y], [max_x, max_y]]
    else:
        # 如果点数不大于二，直接返回原始点
        return points

def get_face_to_target(face_json, target_json):

    # 读取JSON 数据
    with open(face_json, 'r') as f:
        face_data = json.load(f)

    # 读取目标 JSON 数据
    with open(target_json, 'r',encoding='utf-8') as f:
        target_data = json.load(f)

    # 提取 LabelMe 数据中的 shapes 信息
    shapes = face_data.get('shapes', [])

    # 遍历 shapes，提取需要的信息
    for shape in shapes:
        label = shape.get('label', '')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', '')

        # 将提取的信息添加到目标格式的 dataList 中
        target_data['dataList'].append({
            "shapeType": shape_type,
            "label": label,
            "superLabel": "",
            "coordinates": convert_points_to_two(points),
            "properties": {}
        })

    # 将修改后的目标格式数据写入文件
    with open(target_json, 'w',encoding='gbk') as f:
        json.dump(target_data, f, indent=4)

    print(f"get {face_json} to {target_json}")

def batch_convert_labelme_to_target(labelme_dir, target_dir):
    # 遍历 LabelMe JSON 文件所在的目录
    for filename in os.listdir(labelme_dir):
        if filename.endswith('.json'):
            labelme_json = os.path.join(labelme_dir, filename)
            target_json = os.path.join(target_dir, filename)
            get_face_to_target(labelme_json, target_json)

# 示例调用
# 替换为 face anno LabelMe JSON 文件所在目录
# 替换为目标格式的 JSON 文件所在目录
#labelme_dir = "D:/tempdata/cvte_va_headdet/1.testfaceanno"  #test
#target_dir = "D:/tempdata/cvte_va_headdet/1.test"
#labelme_dir = "D:/tempdata/cvte_va_headdet/face_anno/3.202308_headdet_2955"  #3.
#target_dir =  "D:/tempdata/cvte_va_headdet/3.202308_headdet_2955"  
#labelme_dir = "D:/tempdata/cvte_va_headdet/face_anno/4.202308_headdet_30000"  #4.
#target_dir = "D:/tempdata/cvte_va_headdet/4.202308_headdet_30000" 
labelme_dir = "D:/tempdata/cvte_va_headdet/face_anno/5.traindata_offset6pixes_6131"  #5.
target_dir = "D:/tempdata/cvte_va_headdet/5.traindata_offset6pixes_6131" 
batch_convert_labelme_to_target(labelme_dir, target_dir)