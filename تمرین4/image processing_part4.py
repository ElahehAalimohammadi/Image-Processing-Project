# Q4
# نکتهٔ اصلی این است که توان را روی مقادیر خام ۰ تا ۲۵۵ نمی‌زنیم.
# مثلاً 255^1.2 حدود ۸۰۰ می‌شود و بعد از clip بیشتر پیکسل‌ها سفید می‌شوند.
# برای همین اول تصویر را بر ۲۵۵ تقسیم می‌کنیم تا در بازهٔ ۰ تا ۱ باشد، توان می‌زنیم و دوباره در ۲۵۵ ضرب می‌کنیم.
# در گزارش این را به‌عنوان نرمال‌سازی قبل از تبدیل توضیح دهید.
import os
import cv2
import numpy as np

input_folder = "Input_Images"
output_folder = "Output_Images/Part_4"
gama = [0.8, 0.9, 1.1, 1.2]
def apply_gamma(image, g):
    norm = image.astype(np.float32) / 255.0
    tmp = np.power(norm, g) * 255.0
    tmp = np.clip(tmp, 0, 255)
    return np.round(tmp).astype(np.uint8)

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    for g in gama:
        result = apply_gamma(img, g)
        name, ext = os.path.splitext(filename)
        output_name = f"{name}_gamma_{g}{ext}"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, result)
        print("Saved to:", output_path)
















# Elaheh Aali
# Q5
# اول هیستوگرام تصویر ورودی رو حساب میکنیم و ذخیره میکنیم.
# بعد Histogram Equalization رو خودمون پیاده میکنیم:
#   1) هیستوگرام h(k) برای k از 0 تا 255
#   2) هیستوگرام نرمال شده p(k) = h(k) / N
#   3) تابع توزیع تجمعی cdf(k) = جمع p(j) برای j تا k
#   4) نگاشت جدید s(k) = round(255 * cdf(k))
# در آخر هر پیکسل با مقدار k رو با s(k) عوض میکنیم.
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

input_folder = "Input_Images"
output_folder = "Output_Images/Part_5"
os.makedirs(output_folder, exist_ok=True)


def compute_histogram(image):
    # تعداد پیکسلهای هر مقدار از 0 تا 255
    return np.bincount(image.ravel(), minlength=256)


def save_histogram(hist, title, path):
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(256), hist, width=1.0, color="gray")
    plt.xlim(0, 255)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def equalize(image, hist):
    n = image.size
    pdf = hist / n
    cdf = np.cumsum(pdf)
    mapping = np.round(255 * cdf).astype(np.uint8)
    # جدول نگاشت رو یکجا روی همه پیکسلها اعمال میکنیم
    return mapping[image]


for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    print(filename)
    name, ext = os.path.splitext(filename)

    # هیستوگرام تصویر اصلی
    hist = compute_histogram(img)
    save_histogram(hist, f"Histogram of {name}",
                   os.path.join(output_folder, f"{name}_hist.png"))

    # equalization
    eq = equalize(img, hist)
    eq_name = f"{name}_equalized{ext}"
    ok = cv2.imwrite(os.path.join(output_folder, eq_name), eq)
    print("Saved:", ok, eq_name)

    # هیستوگرام تصویر equalize شده
    hist_eq = compute_histogram(eq)
    save_histogram(hist_eq, f"Histogram of {name} (equalized)",
                   os.path.join(output_folder, f"{name}_equalized_hist.png"))

    print("Std before:", round(float(img.std()), 2), " Std after:", round(float(eq.std()), 2))

print(os.path.abspath(output_folder))



























