import os
import argparse
from pathlib import Path

# 定义 VisDrone 类别映射（仅保留车辆相关类别，适配小目标车辆识别）
# 原始类别：1-行人 2-自行车 3-汽车 4-摩托车 5-公交车 6-卡车 7-三轮车 8-遮阳伞 9-垃圾桶 10-灯柱
# 过滤后仅保留车辆类，统一归类为 "vehicle"（class_id=0），也可按需求拆分
VISDRONE_TO_YOLO = {
    3: 0,  # 汽车
    4: 0,  # 摩托车
    5: 0,  # 公交车
    6: 0,  # 卡车
    7: 0  # 三轮车
}


def convert_visdrone_to_yolo(visdrone_anno_path, yolo_anno_path, img_width, img_height):
    """
    转换单张图片的标注文件
    :param visdrone_anno_path: 原始 VisDrone 标注文件路径
    :param yolo_anno_path: 输出 YOLO 标注文件路径
    :param img_width: 图片宽度
    :param img_height: 图片高度
    """
    with open(visdrone_anno_path, 'r', encoding='utf-8') as f_in, \
            open(yolo_anno_path, 'w', encoding='utf-8') as f_out:

        for line in f_in.readlines():
            line = line.strip()
            if not line:
                continue

            # 解析 VisDrone 标注行
            parts = line.split(',')
            if len(parts) < 8:
                continue  # 跳过格式错误的行

            x1 = int(parts[0])
            y1 = int(parts[1])
            w = int(parts[2])
            h = int(parts[3])
            score = int(parts[4])  # score=0 为人工标注目标
            category = int(parts[5])

            # 过滤条件：仅保留标注目标 + 车辆类别 + 有效边界框
            if score != 0 or category not in VISDRONE_TO_YOLO or w <= 0 or h <= 0:
                continue

            # 计算 YOLO 归一化坐标
            x_center = (x1 + w / 2) / img_width
            y_center = (y1 + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            # 过滤极端小目标（可选：如宽/高 < 10 像素的目标，避免无效标注）
            if w < 10 or h < 10:
                continue  # 可根据需求调整阈值

            # 写入 YOLO 标注
            class_id = VISDRONE_TO_YOLO[category]
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def batch_convert(input_dir, output_dir):
    """
    批量转换数据集标注
    :param input_dir: VisDrone 数据集根目录（含 images/ 和 annotations/ 文件夹）
    :param output_dir: 输出 YOLO 格式数据集根目录
    """
    # 创建输出目录结构
    img_output_dir = Path(output_dir) / "images"
    anno_output_dir = Path(output_dir) / "labels"
    img_output_dir.mkdir(parents=True, exist_ok=True)
    anno_output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有标注文件
    anno_dir = Path(input_dir) / "annotations"
    img_dir = Path(input_dir) / "images"
    anno_files = list(anno_dir.glob("*.txt"))

    if not anno_files:
        raise FileNotFoundError(f"未在 {anno_dir} 找到标注文件")

    for anno_file in anno_files:
        # 匹配对应的图片文件（.jpg/.png）
        img_name = anno_file.stem + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            img_path = img_dir / (anno_file.stem + ".png")
            if not img_path.exists():
                print(f"警告：未找到 {anno_file.stem} 对应的图片，跳过")
                continue

        # 获取图片尺寸（无需读取整张图片，提升效率）
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except ImportError:
            print("警告：未安装 PIL，需手动输入图片尺寸（默认 1920x1080）")
            img_width, img_height = 1920, 1080

        # 生成输出路径
        yolo_anno_path = anno_output_dir / anno_file.name
        convert_visdrone_to_yolo(anno_file, yolo_anno_path, img_width, img_height)

        # 复制图片到输出目录（可选：也可创建软链接）
        import shutil
        shutil.copy(img_path, img_output_dir / img_path.name)

    print(f"转换完成！共处理 {len(anno_files)} 个标注文件")
    print(f"YOLO 格式图片：{img_output_dir}")
    print(f"YOLO 格式标注：{anno_output_dir}")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="VisDrone-DET 转 YOLO 格式（适配小目标车辆识别）")
    parser.add_argument("--input", required=True, help="VisDrone 数据集目录（含 images/ 和 annotations/）")
    parser.add_argument("--output", required=True, help="输出 YOLO 格式数据集目录")
    args = parser.parse_args()

    # 执行批量转换
    batch_convert(args.input, args.output)