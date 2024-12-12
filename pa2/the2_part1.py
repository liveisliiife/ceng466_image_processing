import numpy as np
import cv2
from skimage import filters
import os


output_folder = 'THE1_Outputs/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def read_image(filename, gray_scale=False):
    if gray_scale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return img
    img = cv2.imread(filename)
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    cv2.imwrite(output_folder+filename, img)

foto1 = read_image('a1.png')

foto2 = read_image('a2.png')

"""### Question 1 - Pattern Extraction"""

# Convert these images to grayscale

foto1_gray = cv2.cvtColor(foto1, cv2.COLOR_RGB2GRAY)

write_image(foto1_gray, "foto1_gray.jpg")


foto1_array = np.array(foto1)
foto1_array[0][0]

foto1_gray_array = np.array(foto1_gray)
foto1_gray_array[0][0]

foto2_gray = cv2.cvtColor(foto2, cv2.COLOR_RGB2GRAY)

write_image(foto2_gray, "foto2_gray.jpg")


# Apply edge detection filters on the images, use Sobel, Roberts and Prewitt filters.

# sobel filter a1
sobel_x = cv2.Sobel(foto1_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(foto1_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_foto1 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_foto1, "sobel_foto1.jpg")


# sobel filter a2

sobel_x = cv2.Sobel(foto2_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(foto2_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_foto2 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_foto2, "sobel_foto2.jpg")


# roberts filter a1

roberts_foto1 = filters.roberts(foto1_gray)

write_image(roberts_foto1, "roberts_foto1.jpg")


# roberts filter a2

roberts_foto2 = filters.roberts(foto2_gray)

write_image(roberts_foto2, "roberts_foto2.jpg")


# prewitt filter a1

prewitt_foto1 = filters.prewitt(foto1_gray)

write_image(prewitt_foto1, "prewitt_foto1.jpg")


# prewitt filter a2

prewitt_foto2 = filters.prewitt(foto2_gray)

write_image(prewitt_foto2, "prewitt_foto2.jpg")


# 3. Blur the images using three different kernel sizes.

def apply_blur(image, kernel_size=(3, 3)):
    """
    Apply Gaussian blur to the image using the given kernel size.

    Parameters:
    - image: The input grayscale image.
    - kernel_size: The size of the kernel for blurring (e.g., (3, 3), (5, 5)).

    Returns:
    - blurred_image: The resulting blurred image.
    """
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred_image

blurred_foto1_3x3 = apply_blur(foto1_gray, kernel_size=(3, 3))

write_image(blurred_foto1_3x3, "blurred_foto1_3x3.jpg")


blurred_foto1_5x5 = apply_blur(foto1_gray, kernel_size=(5, 5))

write_image(blurred_foto1_5x5, "blurred_foto1_5x5.jpg")


blurred_foto1_7x7 = apply_blur(foto1_gray, kernel_size=(7, 7))

write_image(blurred_foto1_7x7, "blurred_foto1_7x7.jpg")


blurred_foto2_3x3 = apply_blur(foto2_gray, kernel_size=(3, 3))

write_image(blurred_foto2_3x3, "blurred_foto2_3x3.jpg")


blurred_foto2_5x5 = apply_blur(foto2_gray, kernel_size=(5, 5))

write_image(blurred_foto2_5x5, "blurred_foto2_5x5.jpg")


blurred_foto2_7x7 = apply_blur(foto2_gray, kernel_size=(7, 7))

write_image(blurred_foto2_7x7, "blurred_foto2_7x7.jpg")


# 4. Apply Step 2 for each blurred image.

# sobel filter with blurred_foto1_3x3

sobel_x = cv2.Sobel(blurred_foto1_3x3, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_foto1_3x3, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_foto1_3x3 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_foto1_3x3, "sobel_blurred_foto1_3x3.jpg")


# sobel filter with blurred_foto1_5x5

sobel_x = cv2.Sobel(blurred_foto1_5x5, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_foto1_3x3, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_foto1_5x5 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_foto1_5x5, "sobel_blurred_foto1_5x5.jpg")


# sobel filter with blurred_foto1_7x7

sobel_x = cv2.Sobel(blurred_foto1_7x7, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_foto1_7x7, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_foto1_7x7 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_foto1_7x7, "sobel_blurred_foto1_7x7.jpg")


# roberts filter blurred_foto1_3x3

roberts_blurred_foto1_3x3 = filters.roberts(blurred_foto1_3x3)

write_image(roberts_blurred_foto1_3x3, "roberts_blurred_foto1_3x3.jpg")


# roberts filter blurred_foto1_5x5

roberts_blurred_foto1_5x5 = filters.roberts(blurred_foto1_5x5)

write_image(roberts_blurred_foto1_5x5, "roberts_blurred_foto1_5x5.jpg")


# roberts filter blurred_foto1_7x7

roberts_blurred_foto1_7x7 = filters.roberts(blurred_foto1_7x7)

write_image(roberts_blurred_foto1_7x7, "roberts_blurred_foto1_7x7.jpg")


# prewitt filter blurred_foto1_3x3

prewitt_blurred_foto1_3x3 = filters.prewitt(blurred_foto1_3x3)

write_image(prewitt_blurred_foto1_3x3, "prewitt_blurred_foto1_3x3.jpg")


# prewitt filter blurred_foto1_5x5

prewitt_blurred_foto1_5x5 = filters.prewitt(blurred_foto1_5x5)

write_image(prewitt_blurred_foto1_5x5, "prewitt_blurred_foto1_5x5.jpg")



# prewitt filter blurred_foto1_7x7

prewitt_blurred_foto1_7x7 = filters.prewitt(blurred_foto1_7x7)

write_image(prewitt_blurred_foto1_7x7, "prewitt_blurred_foto1_7x7.jpg")



# sobel filter with blurred_foto2_3x3

sobel_x = cv2.Sobel(blurred_foto2_3x3, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_foto2_3x3, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_foto2_3x3 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_foto2_3x3, "sobel_blurred_foto2_3x3.jpg")



# sobel filter with blurred_foto2_5x5

sobel_x = cv2.Sobel(blurred_foto2_5x5, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_foto2_5x5, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_foto2_5x5 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_foto2_5x5, "sobel_blurred_foto2_5x5.jpg")



# sobel filter with blurred_foto2_7x7

sobel_x = cv2.Sobel(blurred_foto2_7x7, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_foto2_7x7, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_foto2_7x7 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_foto2_7x7, "sobel_blurred_foto2_7x7.jpg")



# roberts filter blurred_foto2_3x3

roberts_blurred_foto2_3x3 = filters.roberts(blurred_foto2_3x3)

write_image(roberts_blurred_foto2_3x3, "roberts_blurred_foto2_3x3.jpg")



# roberts filter blurred_foto2_5x5

roberts_blurred_foto2_5x5= filters.roberts(blurred_foto2_5x5)

write_image(roberts_blurred_foto2_5x5, "roberts_blurred_foto2_5x5.jpg")



# roberts filter blurred_foto2_7x7

roberts_blurred_foto2_7x7 = filters.roberts(blurred_foto2_7x7)

write_image(roberts_blurred_foto2_7x7, "roberts_blurred_foto2_7x7.jpg")



# prewitt filter blurred_foto2_3x3

prewitt_blurred_foto2_3x3 = filters.prewitt(blurred_foto2_3x3)

write_image(prewitt_blurred_foto2_3x3, "prewitt_blurred_foto2_3x3.jpg")



# prewitt filter blurred_foto2_5x5

prewitt_blurred_foto2_5x5 = filters.prewitt(blurred_foto2_5x5)

write_image(prewitt_blurred_foto2_5x5, "prewitt_blurred_foto2_5x5.jpg")



# prewitt filter blurred_foto2_7x7

prewitt_blurred_foto2_7x7 = filters.prewitt(blurred_foto2_7x7)

write_image(prewitt_blurred_foto2_7x7, "prewitt_blurred_foto2_7x7.jpg")


# 5. Binarize the gray scale images by extracting the most significant bit.

def binarize_msb(image):
    """
    Binarize the grayscale image by extracting the most significant bit (MSB).

    Parameters:
    - image: The input grayscale image.

    Returns:
    - binarized_image: The resulting binarized image.
    """
    # Extract the most significant bit
    msb = np.right_shift(image, 7)  # Shift the bits right by 7 to get the MSB
    binarized_image = msb * 255  # Multiply by 255 to make the MSB either 0 or 255

    return binarized_image.astype(np.uint8)  # Ensure the output is in uint8 format

# Binarize the images by extracting the MSB
binarized_foto1 = binarize_msb(foto1_gray)

write_image(binarized_foto1, "binarized_foto1.jpg")


# Binarize the images by extracting the MSB
binarized_foto2 = binarize_msb(foto2_gray)

write_image(binarized_foto2, "binarized_foto2.jpg")


# 6. Apply Step 2 for MSB images.

# sobel filter with binarized_foto1

sobel_x = cv2.Sobel(binarized_foto1, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(binarized_foto1, cv2.CV_64F, 0, 1, ksize=3)
sobel_binarized_foto1 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_binarized_foto1, "sobel_binarized_foto1.jpg")


# roberts filter binarized_foto1

roberts_binarized_foto1= filters.roberts(binarized_foto1)

write_image(roberts_binarized_foto1, "roberts_binarized_foto1.jpg")


# prewitt filter binarized_foto1

prewitt_binarized_foto1 = filters.prewitt(binarized_foto1)

write_image(prewitt_binarized_foto1, "prewitt_binarized_foto1.jpg")


# sobel filter with binarized_foto2

sobel_x = cv2.Sobel(binarized_foto2, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(binarized_foto2, cv2.CV_64F, 0, 1, ksize=3)
sobel_binarized_foto2 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_binarized_foto2, "sobel_binarized_foto2.jpg")


# roberts filter binarized_foto2

roberts_binarized_foto2= filters.roberts(binarized_foto2)

write_image(roberts_binarized_foto2, "roberts_binarized_foto2.jpg")


# prewitt filter binarized_foto2

prewitt_binarized_foto2 = filters.prewitt(binarized_foto2)

write_image(prewitt_binarized_foto2, "prewitt_binarized_foto2.jpg")


# 7. Blur the MSB images using three different kernel sizes.

blurred_binarized_foto1_3x3 = apply_blur(binarized_foto1, kernel_size=(3, 3))

write_image(blurred_binarized_foto1_3x3, "blurred_binarized_foto1_3x3.jpg")


blurred_binarized_foto1_5x5 = apply_blur(binarized_foto1, kernel_size=(5, 5))

write_image(blurred_binarized_foto1_5x5, "blurred_binarized_foto1_5x5.jpg")


blurred_binarized_foto1_7x7 = apply_blur(binarized_foto1, kernel_size=(7, 7))

write_image(blurred_binarized_foto1_7x7, "blurred_binarized_foto1_7x7.jpg")


blurred_binarized_foto2_3x3 = apply_blur(binarized_foto2, kernel_size=(3, 3))

write_image(blurred_binarized_foto2_3x3, "blurred_binarized_foto2_3x3.jpg")


blurred_binarized_foto2_5x5 = apply_blur(binarized_foto2, kernel_size=(5, 5))

write_image(blurred_binarized_foto2_5x5, "blurred_binarized_foto2_5x5.jpg")



blurred_binarized_foto2_7x7 = apply_blur(binarized_foto2, kernel_size=(7, 7))

write_image(blurred_binarized_foto2_7x7, "blurred_binarized_foto2_7x7.jpg")



# 8. Apply Step 2 for blurred MSB images.

# sobel filter with blurred_binarized_foto1_3x3

sobel_x = cv2.Sobel(blurred_binarized_foto1_3x3, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_binarized_foto1_3x3, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_binarized_foto1_3x3 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_binarized_foto1_3x3, "sobel_blurred_binarized_foto1_3x3.jpg")


# sobel filter with blurred_binarized_foto1_5x5

sobel_x = cv2.Sobel(blurred_binarized_foto1_5x5, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_binarized_foto1_5x5, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_binarized_foto1_5x5 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_binarized_foto1_5x5, "sobel_blurred_binarized_foto1_5x5.jpg")


# sobel filter with blurred_binarized_foto1_7x7

sobel_x = cv2.Sobel(blurred_binarized_foto1_7x7, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_binarized_foto1_7x7, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_binarized_foto1_7x7 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_binarized_foto1_7x7, "sobel_blurred_binarized_foto1_7x7.jpg")


# roberts filter blurred_binarized_foto1_3x3

roberts_blurred_binarized_foto1_3x3 = filters.roberts(blurred_binarized_foto1_3x3)

write_image(roberts_blurred_binarized_foto1_3x3, "roberts_blurred_binarized_foto1_3x3.jpg")


# roberts filter blurred_binarized_foto1_5x5

roberts_blurred_binarized_foto1_5x5 = filters.roberts(blurred_binarized_foto1_5x5)

write_image(roberts_blurred_binarized_foto1_5x5, "roberts_blurred_binarized_foto1_5x5.jpg")


# roberts filter blurred_binarized_foto1_7x7

roberts_blurred_binarized_foto1_7x7 = filters.roberts(blurred_binarized_foto1_7x7)

write_image(roberts_blurred_binarized_foto1_7x7, "roberts_blurred_binarized_foto1_7x7.jpg")


# prewitt filter blurred_binarized_foto1_3x3

prewitt_blurred_binarized_foto1_3x3 = filters.prewitt(blurred_binarized_foto1_3x3)

write_image(prewitt_blurred_binarized_foto1_3x3, "prewitt_blurred_binarized_foto1_3x3.jpg")


# prewitt filter blurred_binarized_foto1_5x5

prewitt_blurred_binarized_foto1_5x5 = filters.prewitt(blurred_binarized_foto1_5x5)

write_image(prewitt_blurred_binarized_foto1_5x5, "prewitt_blurred_binarized_foto1_5x5.jpg")


# prewitt filter blurred_binarized_foto1_7x7

prewitt_blurred_binarized_foto1_7x7 = filters.prewitt(blurred_binarized_foto1_7x7)

write_image(prewitt_blurred_binarized_foto1_7x7, "prewitt_blurred_binarized_foto1_7x7.jpg")


# sobel filter with blurred_binarized_foto2_3x3

sobel_x = cv2.Sobel(blurred_binarized_foto2_3x3, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_binarized_foto2_3x3, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_binarized_foto2_3x3 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_binarized_foto2_3x3, "sobel_blurred_binarized_foto2_3x3.jpg")


# sobel filter with blurred_binarized_foto2_5x5

sobel_x = cv2.Sobel(blurred_binarized_foto2_5x5, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_binarized_foto2_5x5, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_binarized_foto2_5x5 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_binarized_foto2_5x5, "sobel_blurred_binarized_foto2_5x5.jpg")


# sobel filter with blurred_binarized_foto2_7x7

sobel_x = cv2.Sobel(blurred_binarized_foto2_7x7, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_binarized_foto2_7x7, cv2.CV_64F, 0, 1, ksize=3)
sobel_blurred_binarized_foto2_7x7 = cv2.magnitude(sobel_x, sobel_y)

write_image(sobel_blurred_binarized_foto2_7x7, "sobel_blurred_binarized_foto2_7x7.jpg")


# roberts filter blurred_binarized_foto2_3x3

roberts_blurred_binarized_foto2_3x3 = filters.roberts(blurred_binarized_foto2_3x3)

write_image(roberts_blurred_binarized_foto2_3x3, "roberts_blurred_binarized_foto2_3x3.jpg")


# roberts filter blurred_binarized_foto2_5x5

roberts_blurred_binarized_foto2_5x5 = filters.roberts(blurred_binarized_foto2_5x5)

write_image(roberts_blurred_binarized_foto2_5x5, "roberts_blurred_binarized_foto2_5x5.jpg")


# roberts filter blurred_binarized_foto2_7x7

roberts_blurred_binarized_foto2_7x7 = filters.roberts(blurred_binarized_foto2_7x7)

write_image(roberts_blurred_binarized_foto2_7x7, "roberts_blurred_binarized_foto2_7x7.jpg")


# prewitt filter blurred_binarized_foto2_3x3

prewitt_blurred_binarized_foto2_3x3 = filters.prewitt(blurred_binarized_foto2_3x3)

write_image(prewitt_blurred_binarized_foto2_3x3, "prewitt_blurred_binarized_foto2_3x3.jpg")


# prewitt filter blurred_binarized_foto2_5x5

prewitt_blurred_binarized_foto2_5x5 = filters.prewitt(blurred_binarized_foto2_5x5)

write_image(prewitt_blurred_binarized_foto2_5x5, "prewitt_blurred_binarized_foto2_5x5.jpg")


# prewitt filter blurred_binarized_foto2_7x7

prewitt_blurred_binarized_foto2_7x7 = filters.prewitt(blurred_binarized_foto2_7x7)

write_image(prewitt_blurred_binarized_foto2_7x7, "prewitt_blurred_binarized_foto2_7x7.jpg")


