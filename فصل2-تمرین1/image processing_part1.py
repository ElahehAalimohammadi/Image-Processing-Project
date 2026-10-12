# Q0,1
import os
import cv2
import numpy as np

input_folder = "Input_Images"
output_folder = "Output_Images/Part_1"

def central_energy_ratio(F, factor):
    power = np.abs(F)**2
    n, m = power.shape
    cn, cm = n//2, m//2
    hn, hm = n//(2*factor), m//(2*factor)

    inner = power[cn - hn:cn +hn, cm - hm:cm + hm].sum()
    outer = power.sum() - inner
    return inner/outer

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue
    # تبدیل به 8 بیتی بدون علامت
    image = image.astype(np.uint8)
    print(filename)

    F = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))

    mag = np.log1p(np.abs(F))
    mag = 255 * (mag - mag.min()) / (mag.max()-mag.min())
    mag = mag.astype(np.uint8)

    name, ext = os.path.splitext(filename)
    spectrum_name = name + "_fft" + ext
    spectrum_path = os.path.join(output_folder, spectrum_name)
    cv2.imwrite(spectrum_path, mag)

    ratio_4 = central_energy_ratio(F, 2)
    ratio_16 = central_energy_ratio(F, 4)

    print("Ratio (rate 4):", round(float(ratio_4), 4))
    print("Ratio (rate 16):", round(float(ratio_16), 4))
    print("Saved:", spectrum_name)
print(os.path.abspath(output_folder))