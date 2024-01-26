from glob import glob
import cv2
from PIL import Image  # type: ignore
from pathlib import Path
from typing import List, Tuple
import math
import numpy as np

dpi = 300
width_mm = 210
height_mm = 297
item_mm = 15
gap_h_mm = 5
gap_v_mm = 5
outline = True


def mm_to_inch(mm):
    return mm / 25.4


width_inch = mm_to_inch(width_mm)
height_inch = mm_to_inch(height_mm)
item_inch = mm_to_inch(item_mm)
gap_h_inch = mm_to_inch(gap_h_mm)
gap_v_inch = mm_to_inch(gap_v_mm)


width_pix, height_pix = int(width_inch * dpi), int(height_inch * dpi)
item_w_pix, item_h_pix = int(item_inch * dpi), int(item_inch * dpi)
gap_h_pix, gap_v_pix = int(gap_h_inch * dpi), int(gap_v_inch * dpi)

row_item_num = int((width_pix - 2 * gap_h_pix) / (item_w_pix + gap_h_pix))
col_item_num = int((height_pix - 2 * gap_v_pix) / (item_h_pix + gap_v_pix))
item_per_page = row_item_num * col_item_num

all_imgs_path: List[Path] = [
    file for ext in ["*.jpg", "*.png"] for file in Path(".").rglob(ext)
]
all_imgs: List[np.ndarray] = []

for path in all_imgs_path:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (item_w_pix, item_h_pix), None)
    all_imgs.append(img)

page_num = math.ceil(len(all_imgs) / item_per_page)
print(f"page_num : {page_num}")
print(f"iter_per_page : {item_per_page}")
print(f"row-item: {row_item_num}")
print(f"col-item: {col_item_num}")


def blit(frame: np.ndarray, data: np.ndarray, pos: Tuple[int, int], outline: bool):
    x, y = pos
    row, col = y, x
    w, h = data.shape[:2]
    frame[row : row + w, col : col + h, :] = data
    if outline:
        w, h, c = data.shape
        thickness = int(3 * (dpi / 92))
        contour = np.array(
            [
                (x - thickness, y - thickness),
                (x + w + thickness, y - thickness),
                (x + w + thickness, y + h + thickness),
                (x - thickness, y + h + thickness),
            ]
        )

        cv2.polylines(
            frame,
            [contour],
            isClosed=True,
            color=(0, 0, 0),
            thickness=thickness,
        )


result_page = []
for _ in range(page_num):
    canvas = np.full((height_pix, width_pix, 3), 255, dtype=np.uint8)
    items = all_imgs[:item_per_page].copy()
    if len(all_imgs) > item_per_page:
        all_imgs = all_imgs[item_per_page + 1 :]
    for i, item in enumerate(items):
        row = i // row_item_num
        col = i % row_item_num
        x = gap_h_pix + int(col * (item_w_pix + gap_h_pix))
        y = gap_v_pix + int(row * (item_h_pix + gap_v_pix))
        blit(canvas, item, (x, y), True)
    result_page.append(canvas.copy())
    cv2.imshow("test", cv2.resize(canvas, None, fx=0.2, fy=0.2))
    cv2.waitKey(0)


for i, canvas in enumerate(result_page):
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(canvas)
    canvas.save(f"{i}.png", dpi=(dpi, dpi))  # type: ignore
