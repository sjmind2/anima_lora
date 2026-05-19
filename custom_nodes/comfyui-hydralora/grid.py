import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arial.ttf")
_FONT = None


def _get_font(size: int = 14):
    global _FONT
    if _FONT is not None:
        try:
            return _FONT.font_variant(size=size)
        except Exception:
            pass
    try:
        _FONT = ImageFont.truetype(_FONT_PATH, size)
        return _FONT.font_variant(size=size)
    except Exception:
        _FONT = ImageFont.load_default()
        return _FONT


def _adjusted_font_size(text, initial_size, max_width):
    font = _get_font(initial_size)
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    if tw > max_width * 0.9:
        return max(int(initial_size * (max_width / tw) * 0.9), 8)
    return initial_size


def _truncate_labels(labels, max_len=42):
    if not labels:
        return labels
    actual_max = max(min(max(len(str(s)) for s in labels), max_len), 24)
    return [s if len(str(s)) <= actual_max else str(s)[:actual_max] + "..." for s in labels]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 5:
        tensor = tensor[:, 0]
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.cpu().numpy().clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray(arr)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode == "L":
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def create_grid(
    images_2d: list,
    x_labels: list,
    y_labels: list,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (0, 0, 0),
    grid_spacing: int = 0,
    y_label_orientation: str = "Horizontal",
) -> torch.Tensor:
    rows = len(images_2d)
    cols = len(images_2d[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return torch.zeros(1, 64, 64, 3)

    x_labels = _truncate_labels(x_labels)
    y_labels = _truncate_labels(y_labels)

    i_width, i_height = images_2d[0][0].size

    border_size_top = max(i_width // 15, 20)

    has_y = len(y_labels) > 0 and any(l for l in y_labels)
    has_x = len(x_labels) > 0 and any(l for l in x_labels)

    if has_y:
        y_label_longest = max(len(str(s)) for s in y_labels)
        y_label_scale = min(y_label_longest + 4, 24) / 24
        if y_label_orientation == "Vertical":
            border_size_left = border_size_top
        else:
            border_size_left = int((min(i_width, i_height) + int(0.2 * abs(i_width - i_height))) * y_label_scale)
    else:
        border_size_left = 0
        y_label_scale = 1.0

    if has_y:
        if y_label_orientation == "Vertical":
            x_offset_initial = border_size_left * 3
            bg_width = cols * i_width + (cols - 1) * grid_spacing + 3 * border_size_left
        else:
            x_offset_initial = border_size_left
            bg_width = cols * i_width + (cols - 1) * grid_spacing + border_size_left
    else:
        x_offset_initial = 0
        bg_width = cols * i_width + (cols - 1) * grid_spacing

    if has_x:
        y_offset_initial = 3 * border_size_top
        bg_height = rows * i_height + (rows - 1) * grid_spacing + 3 * border_size_top
    else:
        y_offset_initial = 0
        bg_height = rows * i_height + (rows - 1) * grid_spacing

    background = Image.new("RGBA", (int(bg_width), int(bg_height)), color=bg_color + (255,))

    y_offset = y_offset_initial

    for row in range(rows):
        x_offset = x_offset_initial

        for col in range(cols):
            img = images_2d[row][col]
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            background.paste(img, (x_offset, y_offset))

            if row == 0 and has_x and col < len(x_labels):
                text = str(x_labels[col])
                initial_fs = max(int(48 * i_width / 512), 12)
                font_size = _adjusted_font_size(text, initial_fs, i_width)
                label_h = int(font_size * 1.5)
                font = _get_font(font_size)
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                label_bg = Image.new("RGBA", (i_width, label_h), color=bg_color + (0,))
                d = ImageDraw.Draw(label_bg)
                text_x = (i_width - tw) // 2
                text_y = (label_h - th) // 2
                d.text((text_x, text_y), text, fill=text_color, font=font)
                available = y_offset_initial - label_h
                label_y = max(available // 2, 0)
                background.alpha_composite(label_bg, (x_offset, label_y))

            if col == 0 and has_y and row < len(y_labels):
                text = str(y_labels[row])
                if y_label_orientation == "Vertical":
                    initial_fs = max(int(48 * i_width / 512), 12)
                    font_size = _adjusted_font_size(text, initial_fs, i_width)
                else:
                    label_area_w = int(border_size_left / y_label_scale) if y_label_scale > 0 else border_size_left
                    initial_fs = max(int(48 * label_area_w / 512), 12)
                    font_size = _adjusted_font_size(text, initial_fs, label_area_w)

                font = _get_font(font_size)
                label_h = int(font_size * 1.2)
                label_bg = Image.new("RGBA", (i_height, label_h), color=bg_color + (0,))
                d = ImageDraw.Draw(label_bg)
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                text_x = (i_height - tw) // 2
                text_y = (label_h - th) // 2
                d.text((text_x, text_y), text, fill=text_color, font=font)

                if y_label_orientation == "Vertical":
                    label_bg = label_bg.rotate(90, expand=True)

                available_x = x_offset - label_bg.width
                label_x = max(available_x // 2, 0)

                if y_label_orientation == "Vertical":
                    label_y = y_offset + (i_height - label_bg.height) // 2
                else:
                    label_y = y_offset + i_height - label_bg.height

                background.alpha_composite(label_bg, (label_x, label_y))

            x_offset += i_width + grid_spacing

        y_offset += i_height + grid_spacing

    return pil_to_tensor(background.convert("RGB"))
