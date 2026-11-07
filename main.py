"""图片信息读取与编辑工具。

依赖 OpenCV：
    pip install opencv-python numpy

运行：
    python3 image_editor.py

功能：
    1. 读取图片信息
    2. 缩放、裁剪、旋转、翻转、灰度化、格式转换
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# 支持保存的格式 -> 扩展名
FORMAT_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "GIF": ".gif",
    "WEBP": ".webp",
    "TIFF": ".tiff",
}
# 扩展名 -> 格式（用于无法从文件头识别时兜底）
EXT_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
FORMAT_ALIASES = {"JPG": "JPEG", "TIF": "TIFF"}


class ImageError(Exception):
    """表示可预期的用户输入或图片处理错误。"""


def normalize_format(text):
    """把用户输入的格式名统一成程序支持的格式名。"""
    fmt = str(text).strip().upper().lstrip(".")
    fmt = FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in FORMAT_EXT:
        raise ImageError(
            "不支持的图片格式：{}。可选格式：{}".format(
                text, "、".join(sorted(FORMAT_EXT))
            )
        )
    return fmt


def detect_format(path):
    """通过文件头识别真实图片格式，无法识别时回退到扩展名。"""
    try:
        with open(path, "rb") as file:
            head = file.read(16)
    except OSError:
        head = b""

    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head.startswith(b"BM"):
        return "BMP"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return EXT_FORMAT.get(Path(path).suffix.lower(), "未知")


def human_size(size):
    """把字节数转换成 KB 或 MB，保留两位小数。"""
    value = float(size)
    if value < 1024 * 1024:
        return "{:.2f} KB".format(value / 1024)
    return "{:.2f} MB".format(value / (1024 * 1024))


def _check_path(path):
    """检查路径是否存在且是普通文件。"""
    p = Path(path).expanduser()
    if not p.exists():
        raise ImageError("文件不存在：{}".format(path))
    if not p.is_file():
        raise ImageError("路径不是文件：{}".format(path))
    return p


def load_image(path):
    """读取图片并返回 OpenCV 图像数组；失败时抛出 ImageError。"""
    p = _check_path(path)
    try:
        # 用 imdecode 读取，能兼容中文/空格路径
        data = np.fromfile(str(p), dtype=np.uint8)
        # IMREAD_UNCHANGED 保留 PNG 等格式的透明通道
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    except (OSError, cv2.error) as exc:
        raise ImageError("读取图片失败：{}（{}）".format(path, exc))

    if img is None:
        raise ImageError("文件不是有效的图片，或使用了不支持的图片格式：{}".format(path))
    return img


def _channel_count(img):
    """返回图片通道数。"""
    return 1 if img.ndim == 2 else img.shape[2]


def _color_mode(channels):
    """把通道数转换成易读的颜色模式说明。"""
    modes = {
        1: "灰度（单通道）",
        2: "灰度+透明（2 通道）",
        3: "彩色（BGR，3 通道）",
        4: "彩色+透明（BGRA，4 通道）",
    }
    return modes.get(channels, "{} 通道".format(channels))


def get_image_info(path):
    """读取并返回图片的基本信息。"""
    p = _check_path(path)
    img = load_image(p)
    height, width = img.shape[:2]
    channels = _channel_count(img)
    file_size = p.stat().st_size
    return {
        "name": p.name,
        "path": str(p.resolve()),
        "format": detect_format(p),
        "width": width,
        "height": height,
        "mode": _color_mode(channels),
        "size_bytes": file_size,
        "size_human": human_size(file_size),
    }


def print_image_info(info):
    """以清晰的中文格式打印图片信息。"""
    print("\n===== 图片信息 =====")
    print("文件名：{}".format(info["name"]))
    print("文件路径：{}".format(info["path"]))
    print("图片格式：{}".format(info["format"]))
    print("宽度和高度：{} x {} 像素".format(info["width"], info["height"]))
    print("颜色模式：{}".format(info["mode"]))
    print("文件大小：{}（{} 字节）".format(info["size_human"], info["size_bytes"]))
    print("====================\n")


def resize_image(img, width=None, height=None):
    """缩放图片；只给宽度或高度时，另一边按原比例计算。"""
    if width is None and height is None:
        raise ImageError("宽度和高度至少需要提供一个。")
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise ImageError("宽度必须是大于 0 的整数。")
    if height is not None and (not isinstance(height, int) or height <= 0):
        raise ImageError("高度必须是大于 0 的整数。")

    old_height, old_width = img.shape[:2]
    if width is None:
        # 只给高度，按原宽高比计算宽度
        width = max(1, round(old_width * height / old_height))
    elif height is None:
        # 只给宽度，按原宽高比计算高度
        height = max(1, round(old_height * width / old_width))

    # 缩小时用区域插值，放大时用线性插值
    if width * height < old_width * old_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interpolation)


def crop_image(img, left, top, right, bottom):
    """按左、上、右、下坐标裁剪；右和下为不包含的边界。"""
    height, width = img.shape[:2]
    values = (left, top, right, bottom)
    if any(not isinstance(v, int) for v in values):
        raise ImageError("裁剪坐标必须是整数。")

    # 先检查是否越界，并给出图片实际尺寸作为参考
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ImageError(
            "裁剪区域越界：图片尺寸为 {}x{}，坐标需满足 "
            "0 <= left < right <= {}，0 <= top < bottom <= {}。".format(
                width, height, width, height
            )
        )
    if left >= right or top >= bottom:
        raise ImageError("裁剪坐标不合法：left 必须小于 right，top 必须小于 bottom。")

    # OpenCV 图像切片顺序是 [top:bottom, left:right]
    return img[top:bottom, left:right].copy()


def rotate_image(img, angle):
    """将图片逆时针旋转 90、180 或 270 度。"""
    if angle not in (90, 180, 270):
        raise ImageError("旋转角度只支持 90、180、270 度。")

    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def flip_image(img, direction):
    """水平或垂直翻转图片。"""
    direction = str(direction).strip().lower()
    if direction in ("h", "horizontal", "水平"):
        return cv2.flip(img, 1)
    if direction in ("v", "vertical", "垂直"):
        return cv2.flip(img, 0)
    raise ImageError("翻转方向不合法，请输入 h（水平）或 v（垂直）。")


def grayscale_image(img):
    """把彩色图片转换为灰度图。"""
    channels = _channel_count(img)
    if channels == 1:
        return img.copy()
    if channels == 2:
        return img[:, :, 0].copy()
    if channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ImageError("不支持 {} 通道图片的灰度化。".format(channels))


def _flatten_alpha(img):
    """把透明通道与白色背景混合，去掉透明通道。"""
    channels = _channel_count(img)
    if channels not in (2, 4):
        return img

    alpha = img[:, :, channels - 1].astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    color = img[:, :, : channels - 1].astype(np.float32)
    white = np.full_like(color, 255.0)
    return (color * alpha + white * (1.0 - alpha)).astype(np.uint8)


def _prepare_for_save(img, fmt):
    """保存前调整颜色通道，处理目标格式不支持的透明通道。"""
    channels = _channel_count(img)
    if fmt in ("JPEG", "BMP", "GIF") and channels in (2, 4):
        # JPEG/BMP/GIF 不适合保留 alpha，先铺白底去掉透明通道
        return _flatten_alpha(img)
    if fmt == "JPEG" and channels not in (1, 3):
        raise ImageError("不支持把 {} 通道图片保存为 JPEG。".format(channels))
    return img


def build_output_path(source_path, suffix, fmt):
    """生成不会覆盖已有文件的新文件路径。"""
    source = Path(source_path)
    ext = FORMAT_EXT[fmt]
    candidate = source.with_name(source.stem + suffix + ext)
    index = 1
    while candidate.exists():
        candidate = source.with_name("{}{}_{}{}".format(source.stem, suffix, index, ext))
        index += 1
    return candidate


def save_image(img, source_path, suffix, target_format):
    """把编辑结果保存为新文件，并返回保存路径。"""
    fmt = normalize_format(target_format)
    output_path = build_output_path(source_path, suffix, fmt)
    prepared = _prepare_for_save(img, fmt)

    params = []
    if fmt == "JPEG":
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif fmt == "PNG":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    try:
        success, buffer = cv2.imencode(FORMAT_EXT[fmt], prepared, params)
    except cv2.error as exc:
        raise ImageError("保存图片失败：{}（{}）".format(output_path, exc))
    if not success:
        raise ImageError("保存图片失败：{}".format(output_path))

    try:
        output_path.write_bytes(buffer.tobytes())
    except OSError as exc:
        # 写入失败时清理可能生成的残缺文件
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise ImageError("保存图片失败：{}（{}）".format(output_path, exc))
    return str(output_path.resolve())


def ask_text(prompt):
    """读取一行文本，并去掉用户可能带入的引号。"""
    return input(prompt).strip().strip('"').strip("'")


def ask_optional_int(prompt):
    """读取可留空的整数；留空返回 None。"""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            print("输入无效，请输入整数，或直接回车留空。")


def ask_required_int(prompt):
    """读取必填整数。"""
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            print("输入无效，请输入整数。")


def do_resize(current_path, current_format):
    print("提示：宽度和高度至少填写一个，留空的一项会按比例自动计算。")
    width = ask_optional_int("目标宽度（像素，可留空）：")
    height = ask_optional_int("目标高度（像素，可留空）：")
    img = load_image(current_path)
    result = resize_image(img, width, height)
    return result, "_resized", current_format


def do_crop(current_path, current_format):
    info = get_image_info(current_path)
    print("当前图片尺寸：{} x {}".format(info["width"], info["height"]))
    left = ask_required_int("左坐标 left：")
    top = ask_required_int("上坐标 top：")
    right = ask_required_int("右坐标 right：")
    bottom = ask_required_int("下坐标 bottom：")
    img = load_image(current_path)
    result = crop_image(img, left, top, right, bottom)
    return result, "_cropped", current_format


def do_rotate(current_path, current_format):
    angle = ask_required_int("旋转角度（90 / 180 / 270，逆时针）：")
    img = load_image(current_path)
    result = rotate_image(img, angle)
    return result, "_rotated{}".format(angle), current_format


def do_flip(current_path, current_format):
    direction = ask_text("翻转方向（h = 水平，v = 垂直）：")
    img = load_image(current_path)
    result = flip_image(img, direction)
    is_horizontal = direction.strip().lower() in ("h", "horizontal", "水平")
    suffix = "_flip_h" if is_horizontal else "_flip_v"
    return result, suffix, current_format


def do_grayscale(current_path, current_format):
    img = load_image(current_path)
    result = grayscale_image(img)
    return result, "_gray", current_format


def do_convert(current_path, current_format):
    target = ask_text("目标格式（如 jpg / png / bmp / webp / tiff / gif）：")
    fmt = normalize_format(target)
    img = load_image(current_path)
    return img, "_converted", fmt


def resolve_editable_format(info):
    """确定编辑结果默认使用的保存格式。"""
    try:
        return normalize_format(info["format"])
    except ImageError:
        # 原格式不在可保存列表中时，统一保存为 PNG
        print("提示：原格式 {} 不支持直接保存，编辑结果将保存为 PNG。".format(info["format"]))
        return "PNG"


def main():
    print("========== 图片信息读取与编辑工具 ==========")
    print("输入 q 可随时退出程序。\n")

    # 支持命令行参数传入路径，也支持交互式输入
    current_path = sys.argv[1].strip() if len(sys.argv) > 1 else ask_text("请输入图片文件路径：")
    if not current_path or current_path.lower() == "q":
        return

    # 先验证输入文件，直到得到有效图片或用户退出
    while True:
        try:
            info = get_image_info(current_path)
            break
        except ImageError as exc:
            print("错误：{}".format(exc))
            current_path = ask_text("请重新输入图片路径（q 退出）：")
            if current_path.lower() == "q":
                return

    current_format = resolve_editable_format(info)
    print_image_info(info)

    while True:
        print("请选择操作：")
        print("  1. 查看图片信息")
        print("  2. 缩放")
        print("  3. 裁剪")
        print("  4. 旋转")
        print("  5. 翻转")
        print("  6. 灰度化")
        print("  7. 格式转换")
        print("  0. 退出")
        choice = ask_text("请输入编号：")

        if choice == "0" or choice.lower() == "q":
            print("已退出。")
            return
        if choice == "1":
            try:
                print_image_info(get_image_info(current_path))
            except ImageError as exc:
                print("错误：{}".format(exc))
            continue
        if choice not in ("2", "3", "4", "5", "6", "7"):
            print("无效编号，请重新选择。\n")
            continue

        try:
            if choice == "2":
                result, suffix, target_format = do_resize(current_path, current_format)
            elif choice == "3":
                result, suffix, target_format = do_crop(current_path, current_format)
            elif choice == "4":
                result, suffix, target_format = do_rotate(current_path, current_format)
            elif choice == "5":
                result, suffix, target_format = do_flip(current_path, current_format)
            elif choice == "6":
                result, suffix, target_format = do_grayscale(current_path, current_format)
            else:
                result, suffix, target_format = do_convert(current_path, current_format)

            saved_path = save_image(result, current_path, suffix, target_format)
            current_path = saved_path
            current_format = normalize_format(target_format)

            print("\n操作成功！")
            print("新文件已保存到：{}".format(saved_path))
            print("接下来会继续基于该新文件进行编辑。\n")
            print_image_info(get_image_info(current_path))
        except ImageError as exc:
            print("操作失败：{}\n".format(exc))
        except Exception as exc:  # 兜底，避免意外异常导致程序直接崩溃
            print("发生未预期错误：{}，本次操作已取消。\n".format(exc))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
