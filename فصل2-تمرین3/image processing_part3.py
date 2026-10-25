# Elaheh AaliM
# Q3 - Anti-aliasing and Downsampling

import cv2
import os

input_folder = "Input_Images"
output_folder = "Output_Images/Part_3"

for filename in os.listdir(input_folder):

    path = os.path.join(input_folder, filename)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    # ابعاد اولیه تصویر
    height, width = image.shape

    name, ext = os.path.splitext(filename)

    print(f"\nImage: {filename}")
    print("Original size:", image.shape)

    # -----------------------------------
    # اعمال فیلتر Anti-aliasing
    # -----------------------------------

    # فیلتر میانگین گیری 3x3
    filtered_image = cv2.blur(image, (3, 3))

    # -----------------------------------
    # Downsampling با نرخ 4
    # -----------------------------------

    small_4 = filtered_image[::4, ::4]

    print("Downsampled by 4:", small_4.shape)

    # بازگرداندن تصویر نرخ 4 با Nearest Neighbour
    nearest_4 = cv2.resize(
        small_4,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_4_nearest{ext}"),
        nearest_4
    )

    # بازگرداندن تصویر نرخ 4 با Bilinear
    bilinear_4 = cv2.resize(
        small_4,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_4_bilinear{ext}"),
        bilinear_4
    )

    # -----------------------------------
    # فیلتر مجدد و Downsampling دوم
    # -----------------------------------

    # یک بار دیگر فیلتر Anti-aliasing
    filtered_image_2 = cv2.blur(small_4, (3, 3))

    # کاهش ابعاد دوباره با نرخ 4
    # نرخ کلی کاهش ابعاد برابر 16 می‌شود
    small_16 = filtered_image_2[::4, ::4]

    print("Downsampled by 16:", small_16.shape)

    # بازگرداندن تصویر نرخ 16 با Nearest Neighbour
    nearest_16 = cv2.resize(
        small_16,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_16_nearest{ext}"),
        nearest_16
    )

    # بازگرداندن تصویر نرخ 16 با Bilinear
    bilinear_16 = cv2.resize(
        small_16,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(
        os.path.join(output_folder, f"{name}_rate_16_bilinear{ext}"),
        bilinear_16
    )

    print("All output images saved.")