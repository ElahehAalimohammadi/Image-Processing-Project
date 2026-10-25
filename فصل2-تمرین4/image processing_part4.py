# Elaheh AaliM
# Q4
import cv2
import os
import numpy as np

input_folder = "Input_Images"
output_folder = "Output_Images/Part_4"

os.makedirs(output_folder, exist_ok=True)

# ضرایب فیلتر داده‌شده در صورت سؤال
kernel = np.array([
    0.0267, -0.0169, -0.0782, 0.2669, 0.6029,
    0.2669, -0.0782, -0.0169, 0.0267
], dtype=np.float32)

for filename in os.listdir(input_folder):

    path = os.path.join(input_folder, filename)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    height, width = image.shape
    name, ext = os.path.splitext(filename)

    # فیلتر کردن تصویر در راستای افقی و عمودی
    filtered_image = cv2.sepFilter2D(
        image, cv2.CV_32F, kernel, kernel
    )

    # ذخیره تصویر پس از فیلتر
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(output_folder, f"{name}_filtered{ext}"),
        filtered_image
    )

    # کاهش اندازه با نرخ 4
    small_4 = filtered_image[::4, ::4]

    # بازگرداندن به ابعاد اولیه با Nearest Neighbor
    nearest_4 = cv2.resize(
        small_4, (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # بازگرداندن به ابعاد اولیه با Bilinear
    bilinear_4 = cv2.resize(
        small_4, (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_4_nearest{ext}"),
        nearest_4
    )
    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_4_bilinear{ext}"),
        bilinear_4
    )

    # اعمال مجدد فیلتر روی تصویر کاهش‌یافته
    filtered_image_2 = cv2.sepFilter2D(
        small_4.astype(np.float32),
        cv2.CV_32F,
        kernel,
        kernel
    )

    filtered_image_2 = np.clip(
        filtered_image_2, 0, 255
    ).astype(np.uint8)

    # کاهش اندازه مجدد با نرخ 4؛ نرخ کلی برابر 16 می‌شود
    small_16 = filtered_image_2[::4, ::4]

    # بازگرداندن به ابعاد اولیه با Nearest Neighbor
    nearest_16 = cv2.resize(
        small_16, (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # بازگرداندن به ابعاد اولیه با Bilinear
    bilinear_16 = cv2.resize(
        small_16, (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_16_nearest{ext}"),
        nearest_16
    )
    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_16_bilinear{ext}"),
        bilinear_16
    )

print("Part 4 completed.")