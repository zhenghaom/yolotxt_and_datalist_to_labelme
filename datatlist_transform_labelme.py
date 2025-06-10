import json
import os
colors = {
    'head': (255, 0, 0),  # 红色
    'face': (0, 255, 0),
    'facemask': (255,0, 0), # 绿色
    'person':(0,0, 0)
}
def convert_to_labelme(input_json, output_json):
    # 读取输入的 JSON 数据
    with open(input_json, 'r',errors='ignore') as f:
        data = json.load(f)

    # 创建 LabelMe 格式的 JSON 数据
    labelme_data = {
        "version": "5.1.1",  # LabelMe 版本号，可根据实际情况修改
        "flags": {},  # 标志字段，通常为空
        "shapes": [],  # 存储标注信息的列表
        "imagePath": data["filePath"].split("/")[-1],  # 图像路径，取文件名部分
        "imageData": None,  
        "imageHeight": data["info"]["height"],  # 图像高度
        "imageWidth": data["info"]["width"]  # 图像宽度
    }
    #group_id=0
    # 遍历输入数据中的 dataList，转换为 LabelMe 的 shapes 格式
    for item in data["dataList"]:
        # 检查是否存在 points 字段，如果存在则使用 points，否则使用 coordinates
        points_key = 'points' if 'points' in item else 'coordinates'
        shape = {
            "label": item["label"],  # 标注类别
            "confidence":1,
            "points": item[points_key],  # 使用 points 或 coordinates 字段
            "group_id": None,  #后期再写脚本合并人头人脸，更新groupid
            "shape_type": item["shapeType"],  # 形状类型，如 rectangle
            "fill_color": None,
            "line_color":colors[item["label"]],
            "flags": {}  # 标志字段，通常为空
        }
        #group_id=group_id+1
        labelme_data["shapes"].append(shape)
    # 删除原来的 JSON 文件
    os.remove(input_json)
    # 将转换后的数据写入输出文件
    with open(output_json, 'w',encoding='gbk') as f:
        json.dump(labelme_data, f, indent=4)

def batch_convert_to_labelme(directory_path):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件是否是 JSON 文件
        if filename.endswith('.json'):
            input_json = os.path.join(directory_path, filename)
            # 输出文件名
            output_json = os.path.join(directory_path, filename)
            # 转换 JSON 格式
            convert_to_labelme(input_json, output_json)
            print(f"Converted {filename} to {output_json}")
            

# 示例调用
# 替换为你的 JSON 文件所在目录
#directory_path = 'D:/tempdata/cvte_va_headdet/1.test' #test
#directory_path = 'D:/tempdata/cvte_va_headdet/2.202304_headdet_4831'  #2.
#directory_path = "D:/tempdata/cvte_va_headdet/3.202308_headdet_2955"  #3.
#directory_path = "D:/tempdata/cvte_va_headdet/4.202308_headdet_30000"  #4.
directory_path = "D:/tempdata/cvte_va_headdet/5.traindata_offset6pixes_6131" #5.
batch_convert_to_labelme(directory_path)