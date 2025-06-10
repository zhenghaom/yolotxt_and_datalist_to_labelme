import json
import os
from PIL import Image

def yolo_to_datalist(txt_path, image_path, class_map, output_json_path):
    """
    将 YOLO 格式的 TXT 文件转换为 datalist 格式的 JSON 文件。
    :param txt_path: YOLO 格式的 TXT 文件路径
    :param image_path: 对应的图像文件路径
    :param class_map: 类别映射字典，将 YOLO 的类别索引映射到实际的类别名称
    :param output_json_path: 输出的 datalist 格式 JSON 文件路径
    """
    with Image.open(image_path) as img:
        img_width = img.width
        img_height = img.height

    shapes = []
    with open(txt_path, 'r') as file:
        id = 0
        lines = file.readlines()
        for line in lines:
            if line==None:continue
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 将归一化坐标转换为绝对坐标
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            # 计算矩形的左上角和右下角坐标
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 创建 datalist 格式的 shape 对象
            shapes.append({
                'id': id,
                "label": class_map[class_id],
                "coordinates": [[x1, y1], [x2, y2]],
                "shapeType": "rectangle"
            })
            id += 1
    image_path_1=image_path    
    image_path_1 = image_path_1.replace("\\", "/")
    # 创建 datalist 格式的 JSON 数据
    data_list = {
        "filePath": image_path_1,
        "info": {
            "height": img_height,
            "width": img_width,
            "depth": 3
        },
        "dataList": shapes
    }
    
    # 删除原来的 TXT 文件
    os.remove(txt_path)
    
    # 将 JSON 数据写入文件
    with open(output_json_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=2)

def batch_convert_yolo_to_datalist(txt_folder_path, output_folder_path, class_map):
    """
    批量转换 YOLO 格式的 TXT 文件为 datalist 格式的 JSON 文件。
    :param txt_folder_path: YOLO 格式的 TXT 文件所在目录
    :param output_folder_path: 输出的 datalist 格式 JSON 文件所在目录
    :param class_map: 类别映射字典
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for txt_file in os.listdir(txt_folder_path):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_folder_path, txt_file)
            output_json_path = os.path.join(output_folder_path, txt_file.replace('.txt', '.json'))
            image_path = os.path.join(txt_folder_path, txt_file.replace('.txt', '.jpg'))
            yolo_to_datalist(txt_path, image_path, class_map, output_json_path)
            print(f"Converted {txt_file} to {output_json_path}")

# 示例用法
#txt_folder_path = "D:/tempdata/cvte_va_headdet/1.test"  # YOLO 格式的 TXT 文件所在目录
txt_folder_path = "D:/tempdata/cvte_va_headdet/5.traindata_offset6pixes_6131"  # 5.

output_folder_path = txt_folder_path  # 输出的 datalist 格式 JSON 文件所在目录
class_map = {0: 'head'}  # 类别映射字典

batch_convert_yolo_to_datalist(txt_folder_path, output_folder_path, class_map)