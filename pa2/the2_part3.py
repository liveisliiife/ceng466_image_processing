import numpy as np
import pywt
import cv2
import os
from scipy.fftpack import dct, idct
from skimage.metrics import mean_squared_error
# pip install PyWavelets

# Paths to the images
image_paths = ["c1.jpg", "c2.jpg", "c3.jpg"]
output_folder = "compressed_images"
os.makedirs(output_folder, exist_ok=True)

# Function to apply DCT and IDCT
def apply_dct(image, n):
    dct_transformed = dct(dct(image.T, norm='ortho').T, norm='ortho')
    flattened = np.abs(dct_transformed).flatten()
    threshold = np.percentile(flattened, 100 - n)
    dct_transformed[np.abs(dct_transformed) < threshold] = 0
    idct_image = idct(idct(dct_transformed.T, norm='ortho').T, norm='ortho')
    return dct_transformed, idct_image

# Function to apply Haar Wavelet Transform and Inverse
def apply_wavelet(image, n):
    coeffs = pywt.wavedec2(image, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs
    coeffs_flattened = np.abs(np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten())))
    threshold = np.percentile(coeffs_flattened, 100 - n)

    # Apply threshold
    cA[np.abs(cA) < threshold] = 0
    cH[np.abs(cH) < threshold] = 0
    cV[np.abs(cV) < threshold] = 0
    cD[np.abs(cD) < threshold] = 0

    coeffs_compressed = (cA, (cH, cV, cD))
    reconstructed_image = pywt.waverec2(coeffs_compressed, 'haar')
    return coeffs_compressed, reconstructed_image

# Compression levels
compression_levels = [1, 10, 50]
results = {}

# Process each image
for image_path in image_paths:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error reading {image_path}")
        continue

    results[image_name] = {}

    for n in compression_levels:
        # DCT Compression
        dct_compressed, dct_reconstructed = apply_dct(original_image, n)
        dct_mse = mean_squared_error(original_image, dct_reconstructed)
        dct_filename = f"{output_folder}/{image_name}_dct_{n}.jpg"
        cv2.imwrite(dct_filename, np.clip(dct_reconstructed, 0, 255).astype(np.uint8))

        # Wavelet Compression
        wavelet_compressed, wavelet_reconstructed = apply_wavelet(original_image, n)
        wavelet_mse = mean_squared_error(original_image, wavelet_reconstructed)
        wavelet_filename = f"{output_folder}/{image_name}_wavelet_{n}.jpg"
        cv2.imwrite(wavelet_filename, np.clip(wavelet_reconstructed, 0, 255).astype(np.uint8))

        results[image_name][n] = {
            "dct": {
                "mse": dct_mse,
                "size": os.path.getsize(dct_filename)
            },
            "wavelet": {
                "mse": wavelet_mse,
                "size": os.path.getsize(wavelet_filename)
            }
        }

# Display results
for image_name, data in results.items():
    print(f"Results for {image_name}:")
    for n, metrics in data.items():
        print(f"  Compression Level N={n}%:")
        print(f"    DCT -> MSE: {metrics['dct']['mse']:.4f}, Size: {metrics['dct']['size']} bytes")
        print(f"    Wavelet -> MSE: {metrics['wavelet']['mse']:.4f}, Size: {metrics['wavelet']['size']} bytes")
