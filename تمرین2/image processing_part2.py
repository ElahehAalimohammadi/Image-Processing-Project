# Elaheh AaliM
# Q3
# اومدم اینجا ضریب a رو در مقادیر پیکسل ها ضرب کردم که این کار
# هم توی شدت روشنایی اثر میذاره و هم بازهی تغییرات
# رو عوض میکنه و روی کنتراست هم اثر داره.
import os
import numpy as np
import cv2 as cv

def changing_a(image, a):
    tmp = image.astype(np.int16) * a
    tmp = np.clip(tmp, 0, 255)
    return tmp.astype(np.uint8)

input_folder = "input_images"
output_folder = "output_images/part_2"
a_array = [0.5, 0.75, 1.25, 1.5]
for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if image is None:
        continue
    for a in a_array:
        result = changing_a(image,a)
        name, ext = os.path.splitext(filename)
        output_name = f'{name}_{a}{ext}'
        output_path = os.path.join(output_folder,output_name)
        cv.imwrite(output_path, result)
        print("Saved to:", output_name)