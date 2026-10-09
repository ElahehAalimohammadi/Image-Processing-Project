#Elaheh Aali
# Q3
from pickletools import optimize

import numpy as np
import cv2 as cv
import os

input_folder = "Input_Images"
output_folder = "Output_Images/Part_3"
low_p = 1
high_p = 99

def apply_linear(image, a, b):
    tmp = a * image.astype(np.float16) + b
    tmp = np.clip(tmp, 0, 255)
    return tmp.astype(np.uint8)
for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        continue
    print(filename)

    lo, hi = np.percentile(img, [low_p, high_p])
    # print(lo, hi)
    if hi - lo < 1e-6:
        # تصویر تقریبا ثابته و کشیدن کنتراست معنی نداره
        a, b = 1.0, 0.0
    else:
        # میخوایم lo بره روی 0 و hi بره روی 255:
        # a*lo + b = 0  و  a*hi + b = 255
        a = 255.0 / (hi - lo)
        b = -a * lo
    optimized = apply_linear(img, a, b)

    name ,ext = os.path.splitext(filename)
    optimized_name = name + ext
    optimized_path = os.path.join(output_folder, optimized_name)
    cv.imwrite(optimized_path, optimized)

    print("lo, hi:", lo, hi)
    print("Optimal a:", round(a, 4))
    print("Optimal b:", round(b, 2))
    print("Std before:", round(float(img.std()), 2), " Std after:", round(float(optimized.std()), 2))
    print("Saved:", optimized_name)

print(os.path.abspath(output_folder))