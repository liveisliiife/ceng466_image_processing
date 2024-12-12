import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift


'''
def plot_image_and_spectrum_fourier(image, channel_name="Image"):
    """
    This function takes an image and visualizes both the image and its Fourier Spectrum.
    """
    # Show the spatial domain image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'{channel_name} - Spatial Domain')
    plt.axis('off')

    # Fourier Transform and Spectrum
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # Show the frequency domain (Fourier Transform)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'{channel_name} - Fourier Domain')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    

def plot_image_and_spectrum_gaussian(image, channel_name="Image"):
    """
    This function takes an image and visualizes both the image and its Fourier Spectrum.
    """
    # Show the spatial domain image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'{channel_name} - Spatial Domain')
    plt.axis('off')

    # Apply Spatial Filters  ---> GaussianBlur
    blurred_color_channel = cv2.GaussianBlur(image, (5, 5), 0)  # we can use different kernel sizes

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_color_channel, cmap='gray')
    plt.title(f'{channel_name} - GaussianBlur')
    plt.axis('off')
    plt.tight_layout()
    plt.show()




# Visualize the spatial domain and Fourier domain for each channel
plot_image_and_spectrum_fourier(r1_channel, "Red Channel")
plot_image_and_spectrum_fourier(g1_channel, "Green Channel")
plot_image_and_spectrum_fourier(b1_channel, "Blue Channel")



plot_image_and_spectrum_gaussian(r1_channel, "Red Channel")
plot_image_and_spectrum_gaussian(g1_channel, "Green Channel")
plot_image_and_spectrum_gaussian(b1_channel, "Blue Channel")



# Visualize the spatial domain and Fourier domain for each channel
plot_image_and_spectrum_fourier(r2_channel, "Red Channel")
plot_image_and_spectrum_fourier(g2_channel, "Green Channel")
plot_image_and_spectrum_fourier(b2_channel, "Blue Channel")

plot_image_and_spectrum_gaussian(r2_channel, "Red Channel")
plot_image_and_spectrum_gaussian(g2_channel, "Green Channel")
plot_image_and_spectrum_gaussian(b2_channel, "Blue Channel")



# Visualize the spatial domain and Fourier domain for each channel
plot_image_and_spectrum_fourier(r3_channel, "Red Channel")
plot_image_and_spectrum_fourier(g3_channel, "Green Channel")
plot_image_and_spectrum_fourier(b3_channel, "Blue Channel")

'''


img1 = cv2.imread('b1.jpg')

# Separate the RGB channels
b1_channel = img1[:,:,0]  # Blue channel
g1_channel = img1[:,:,1]  # Green channel
r1_channel = img1[:,:,2]  # Red channel



img2 = cv2.imread('b2.jpg')

# Separate the RGB channels
b2_channel = img2[:,:,0]  # Blue channel
g2_channel = img2[:,:,1]  # Green channel
r2_channel = img2[:,:,2]  # Red channel



img3 = cv2.imread('b3.jpg')

# Separate the RGB channels
b3_channel = img3[:,:,0]  # Blue channel
g3_channel = img3[:,:,1]  # Green channel
r3_channel = img3[:,:,2]  # Red channel





def apply_gaussian_filter(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def ideal_low_pass_filter(image, cutoff_frequency):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_frequency:
                mask[i, j] = 1

    dft = fftshift(fft2(image))
    filtered_dft = dft * mask
    filtered_image = np.abs(ifft2(ifftshift(filtered_dft)))
    return np.uint8(filtered_image)

def band_pass_filter(image, low_cutoff, high_cutoff):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if low_cutoff < dist < high_cutoff:
                mask[i, j] = 1

    dft = fftshift(fft2(image))
    filtered_dft = dft * mask
    filtered_image = np.abs(ifft2(ifftshift(filtered_dft)))
    return np.uint8(filtered_image)

def band_reject_filter(image, low_cutoff, high_cutoff):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if low_cutoff < dist < high_cutoff:
                mask[i, j] = 0

    dft = fftshift(fft2(image))
    filtered_dft = dft * mask
    filtered_image = np.abs(ifft2(ifftshift(filtered_dft)))
    return np.uint8(filtered_image)

def process_image(image_path, output_prefix, params_per_channel):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    channels = cv2.split(image)
    gaussian_filtered_channels = []
    median_filtered_channels = []
    ilp_filtered_channels = []
    bp_filtered_channels = []
    br_filtered_channels = []

    for idx, channel in enumerate(channels):
        # Get channel-specific parameters
        gaussian_kernel = params_per_channel[idx]["gaussian_kernel"]
        median_kernel = params_per_channel[idx]["median_kernel"]
        ilp_cutoff = params_per_channel[idx]["ilp_cutoff"]
        bp_low = params_per_channel[idx]["bp_low"]
        bp_high = params_per_channel[idx]["bp_high"]
        br_low = params_per_channel[idx]["br_low"]
        br_high = params_per_channel[idx]["br_high"]

        # Apply spatial domain filters
        gaussian_filtered = apply_gaussian_filter(channel, gaussian_kernel)
        gaussian_filtered_channels.append(gaussian_filtered)
        median_filtered = apply_median_filter(channel, median_kernel)
        median_filtered_channels.append(median_filtered)

        # Save spatial domain results for the channel
        cv2.imwrite(f"{output_prefix}_channel{idx+1}_gaussian.png", gaussian_filtered)
        cv2.imwrite(f"{output_prefix}_channel{idx+1}_median.png", median_filtered)

        # Apply Fourier domain filters
        ilp_filtered = ideal_low_pass_filter(channel, ilp_cutoff)
        ilp_filtered_channels.append(ilp_filtered)
        bp_filtered = band_pass_filter(channel, bp_low, bp_high)
        bp_filtered_channels.append(bp_filtered)
        br_filtered = band_reject_filter(channel, br_low, br_high)
        br_filtered_channels.append(br_filtered)

        # Save Fourier domain results for the channel
        cv2.imwrite(f"{output_prefix}_channel{idx+1}_ilp.png", ilp_filtered)
        cv2.imwrite(f"{output_prefix}_channel{idx+1}_bp.png", bp_filtered)
        cv2.imwrite(f"{output_prefix}_channel{idx+1}_br.png", br_filtered)

    # Merge processed channels back into RGB images and save
    gaussian_rgb = cv2.merge(gaussian_filtered_channels)
    cv2.imwrite(f"{output_prefix}_gaussian_rgb.png", gaussian_rgb)

    median_rgb = cv2.merge(median_filtered_channels)
    cv2.imwrite(f"{output_prefix}_median_rgb.png", median_rgb)

    ilp_rgb = cv2.merge(ilp_filtered_channels)
    cv2.imwrite(f"{output_prefix}_ilp_rgb.png", ilp_rgb)

    bp_rgb = cv2.merge(bp_filtered_channels)
    cv2.imwrite(f"{output_prefix}_bp_rgb.png", bp_rgb)

    br_rgb = cv2.merge(br_filtered_channels)
    cv2.imwrite(f"{output_prefix}_br_rgb.png", br_rgb)

    # Combine all filters for visualization
    final_image = cv2.merge(gaussian_filtered_channels)
    final_image = cv2.addWeighted(final_image, 0.5, cv2.merge(median_filtered_channels), 0.5, 0)
    cv2.imwrite(f"{output_prefix}_final.png", final_image)

# Example usage with channel-specific parameters, but I prefer to use Otsu
params = [
    {"gaussian_kernel": 5, "median_kernel": 5, "ilp_cutoff": 30, "bp_low": 10, "bp_high": 50, "br_low": 20, "br_high": 60},
    {"gaussian_kernel": 7, "median_kernel": 5, "ilp_cutoff": 25, "bp_low": 15, "bp_high": 55, "br_low": 25, "br_high": 65},
    {"gaussian_kernel": 9, "median_kernel": 5, "ilp_cutoff": 20, "bp_low": 20, "bp_high": 60, "br_low": 30, "br_high": 70}
]

process_image("b1.jpg", "b1", params)
process_image("b2.jpg", "b2", params)
process_image("b3.jpg", "b3", params)

















