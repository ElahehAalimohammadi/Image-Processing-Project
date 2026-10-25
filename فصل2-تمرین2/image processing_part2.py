# Elaheh AaliM
# Q2 - Downsampling and Upsampling

import cv2
import os

input_folder = "Input_Images"
output_folder = "Output_Images/Part_2"

os.makedirs(output_folder, exist_ok=True)

# نرخ‌های Downsampling
rates = [4, 16]

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

    # Downsampling با نرخ‌های 4 و 16
    for rate in rates:

        # کاهش ابعاد بدون اعمال پردازش دیگری
        # تعداد گام ها رو ربت تعیین میکنه که هر چندتا گام این اتفاق رخ بده.
        small_image = image[::rate, ::rate]

        print(f"Downsampled by {rate}:", small_image.shape)

        # بازگرداندن تصویر به ابعاد اولیه با Nearest Neighbour
        nearest = cv2.resize(
            small_image,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        nearest_name = f"{name}_rate_{rate}_nearest{ext}"
        nearest_path = os.path.join(output_folder, nearest_name)

        cv2.imwrite(nearest_path, nearest)

        # بازگرداندن تصویر به ابعاد اولیه با Bilinear
        bilinear = cv2.resize(
            small_image,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        bilinear_name = f"{name}_rate_{rate}_bilinear{ext}"
        bilinear_path = os.path.join(output_folder, bilinear_name)

        cv2.imwrite(bilinear_path, bilinear)

        print("Saved:", nearest_name)
        print("Saved:", bilinear_name)# Elaheh AaliM
# Q2 - Downsampling and Upsampling

import cv2
import os

# مسیر پوشه تصاویر
input_folder = "Input_Images"
output_folder = "Output_Images/Part_2"

# ساخت پوشه خروجی در صورت نیاز
os.makedirs(output_folder, exist_ok=True)

# نرخ‌های Downsampling
rates = [4, 16]

# خواندن تصاویر ورودی
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

    # Downsampling با نرخ‌های 4 و 16
    for rate in rates:

        # کاهش ابعاد بدون اعمال پردازش دیگری
        small_image = image[::rate, ::rate]

        print(f"Downsampled by {rate}:", small_image.shape)

        # بازگرداندن تصویر به ابعاد اولیه با Nearest Neighbour
        nearest = cv2.resize(
            small_image,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        nearest_name = f"{name}_rate_{rate}_nearest{ext}"
        nearest_path = os.path.join(output_folder, nearest_name)

        cv2.imwrite(nearest_path, nearest)

        # بازگرداندن تصویر به ابعاد اولیه با Bilinear
        bilinear = cv2.resize(
            small_image,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        bilinear_name = f"{name}_rate_{rate}_bilinear{ext}"
        bilinear_path = os.path.join(output_folder, bilinear_name)

        cv2.imwrite(bilinear_path, bilinear)

        print("Saved:", nearest_name)
        print("Saved:", bilinear_name)