# Elaheh Aali
# Q5
# اول هیستوگرام تصویر ورودی رو حساب میکنیم و ذخیره میکنیم.
# بعد Histogram Equalization رو با هیستوگرام اسموث شده پیاده میکنیم:
#   1) هیستوگرام h(k) برای k از 0 تا 255
#   2) اسموث کردن: هر h(k) رو با میانگین چندتا همسایه اش (k-r تا k+r) عوض میکنیم
#   3) هیستوگرام نرمال شده p(k) = h_smooth(k) / جمع h_smooth
#   4) تابع توزیع تجمعی cdf(k) = جمع p(j) برای j تا k
#   5) نگاشت جدید s(k) = round(255 * cdf(k))
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

# شعاع اسموث: r=2 یعنی میانگین 5 تا (خود مقدار + 2 تا از هر طرف). اگه کتاب عدد دیگه ای گفته اینو عوض کن.
radius = 2


def compute_histogram(image):
    # تعداد پیکسلهای هر مقدار از 0 تا 255
    return np.bincount(image.ravel(), minlength=256)


def smooth_histogram(hist, r):
    # لبه ها رو با مقدار کنارشون پر میکنیم که میانگین گیری کنار 0 و 255 خراب نشه
    padded = np.pad(hist.astype(np.float64), r, mode="edge")
    kernel = np.ones(2 * r + 1) / (2 * r + 1)
    return np.convolve(padded, kernel, mode="valid")


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


def equalize(image, hist, r):
    smooth = smooth_histogram(hist, r)
    pdf = smooth / smooth.sum()
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

    # equalization با هیستوگرام اسموث شده
    eq = equalize(img, hist, radius)
    eq_name = f"{name}_equalized{ext}"
    ok = cv2.imwrite(os.path.join(output_folder, eq_name), eq)
    print("Saved:", ok, eq_name)

    # هیستوگرام تصویر equalize شده
    hist_eq = compute_histogram(eq)
    save_histogram(hist_eq, f"Histogram of {name} (equalized)",
                   os.path.join(output_folder, f"{name}_equalized_hist.png"))

    print("Std before:", round(float(img.std()), 2), " Std after:", round(float(eq.std()), 2))

print(os.path.abspath(output_folder))