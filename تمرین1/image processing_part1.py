# Elaheh AaliM
# Q0
# میخوایم تصویر رو بخونیم و به شکل ماتریس نشون بدیم. مقادیر مانریسا بین 0 تا 255 ئه.
import cv2
import os
import numpy as np

brightnes = [40, 20, -20, -40]
#Q1
#میخوایم روشنایی تصوویر رو تغییر بدیم بریا این کار یه تابع مینویسم که تصویر رو اول 16 تایی نمایش بده که سرریز رخ نده بعد که روشناییش رو تغییر دادم میبرم دوباره به 8 بیتی.

def change_brightness(image, b):
    tmp = image.astype(np.int16) + b
    tmp = np.clip(tmp, 0, 255)
    return tmp.astype(np.uint8)

# مسیر پوشه تصاویر
input_folder = "Input_Images"
output_folder = "Output_Images/Part_1"

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    # تبدیل به 8 بیتی بدون علامت
    image = image.astype(np.uint8)
    print(f"\nImage: {filename}")
    print("Matrix:")
    print(image)
    print("Shape:", image.shape)
    print("Data type:", image.dtype)

    for b in brightnes:
        resualt = change_brightness(image, b)
        name ,ext = os.path.splitext(filename)
        if b>0:
            output_name = f"{name}+{b}{ext}"
        else:
            output_name = f"{name}-{abs(b)}{ext}"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, resualt)
        print("Saved to:", output_name)

    # حالا بیایم ببینیم که مقدار بهینه برای b چی باشه بهتره. اول میانگین شدت روشنایی رو حساب میکنم بهد باید ییه b پیدا کنم که این میانگین رو به 128 نزدیکش کنه.

    mean_intensity = np.mean(image)
    b_opt = round(128 - mean_intensity)
    b_opt = max(-255, min(255, b_opt))
    # تولید تصویر بهینه
    optimized = change_brightness(image, b_opt)

    optimized_name = f"{name}_optimal_b_{b_opt}{ext}"
    optimized_path = os.path.join(output_folder, optimized_name)

    cv2.imwrite(optimized_path, optimized)

    print("Mean intensity:", round(mean_intensity, 2))
    print("Optimal b:", b_opt)
    print("Saved:", optimized_name)


