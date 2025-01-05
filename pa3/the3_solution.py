import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

input_folder = 'THE3_Images/'
output_folder = 'THE3_Outputs/'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def read_image(filename, gray_scale=False):
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE)
        return img
    img = cv2.imread(input_folder + filename)
    return img

def write_image(img, filename):
    cv2.imwrite(output_folder + filename, img)

def display_image(img, title):
    plt.figure(figsize=(6, 6))
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap='gray')
    else:  # RGB
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Preprocessing step to enhance contrast
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)
    return enhanced_img

# Postprocessing step to remove small noise
def postprocess_image(img):
    kernel = np.ones((3, 3), np.uint8)  # we can try (5, 5)
    cleaned_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cleaned_img

# Morphological operations
def grayscale_morphology(img, operation, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == 'erode':
        return cv2.erode(img, kernel, iterations=iterations)
    elif operation == 'dilate':
        return cv2.dilate(img, kernel, iterations=iterations)
    elif operation == 'open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Invalid operation")

# Apply color to clusters (RGB values)
def apply_color_to_clusters(labels, n_clusters):
    colors = np.random.randint(0, 255, size=(n_clusters, 3), dtype=np.uint8)
    colored_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i in range(n_clusters):
        colored_image[labels == i] = colors[i]
    return colored_image

# KMeans Clustering using RGB Features
def kmeans_rgb(img, n_clusters, max_iter=300):  # we can use max_iter=500 but it will need more time
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter)
    labels = kmeans.fit_predict(pixel_values)
    segmented_img = labels.reshape(img.shape[:2])
    colored_segmented_img = apply_color_to_clusters(segmented_img, n_clusters)
    return colored_segmented_img

# KMeans Clustering using LBP Features
def kmeans_lbp(img, n_clusters, radius=2, n_points=8, method='uniform', max_iter=300):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    lbp = np.nan_to_num(lbp, nan=0.0)
    lbp = lbp.astype(np.float32).reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter)
    labels = kmeans.fit_predict(lbp)
    segmented_img = labels.reshape(gray.shape)
    colored_segmented_img = apply_color_to_clusters(segmented_img, n_clusters)
    return colored_segmented_img

# Graph-based Segmentation using GrabCut
def grabcut_segmentation(img, rect=None, iter_count=5):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Use a default rectangle if none is provided
    if rect is None:
        rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10) 
    
    # Apply GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask to create a binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the image
    result = img * mask2[:, :, np.newaxis]
    return result

# Images
hali1 = read_image("1.png")  # a
hali2 = read_image("2.png")  # b
carsaf1 = read_image("3.png")  # c
carsaf2 = read_image("4.png")  # d
fermuar1 = read_image("5.png")  # e
fermuar2 = read_image("6.png")  # f

images = [hali1, hali2, carsaf1, carsaf2, fermuar1, fermuar2]
image_names = ["hali1", "hali2", "carsaf1", "carsaf2", "fermuar1", "fermuar2"]

# Apply and save results
for i, img in enumerate(images):
    # Preprocessing
    preprocessed_img = preprocess_image(img)

    print(" preprocessed has finished")

    # 1. Grayscale Morphology
    for kernel_size in [3,7,11]:   # we can try more value like [3, 5, 7, 9, 11]
        for operation in ['erode', 'dilate', 'open', 'close']:
            for iterations in [1, 3]:    #  we can try more value
                result = grayscale_morphology(preprocessed_img, operation, kernel_size, iterations)
                postprocessed_result = postprocess_image(result)
                write_image(postprocessed_result, f"{image_names[i]}_morph_{operation}_{kernel_size}_iteration_{iterations}.png")

    print(" Grayscale Morphology has finished")

    # 2. KMeans Clustering using RGB
    for n_clusters in [2, 3, 4, 5]:   # we can try more value
        result = kmeans_rgb(img, n_clusters, max_iter=300)  # max_iter=500 ?
        write_image(result, f"{image_names[i]}_kmeans_rgb_{n_clusters}.png")

    print(" KMeans Clustering using RGB has finished")

    # 3. KMeans Clustering using LBP
    for n_clusters in [2, 3, 4, 5]:     # we can try more value
        for radius in [2, 3, 4, 5]:     # we can try more value
            for method in ['uniform', 'default', 'ror', 'var']:
                result = kmeans_lbp(img, n_clusters, radius, method=method, max_iter=300) # max_iter=500 ?
                write_image(result, f"{image_names[i]}_kmeans_lbp_{n_clusters}_r{radius}_{method}.png")
    
    print(" KMeans Clustering using LBP has finished")

    """
    # 4. I want to try Graph Cuts (GrabCut) Segmentation with Iteration 5, we can try 3,5,7
    for iter_count in [5]:
        grabcut_result = grabcut_segmentation(img, iter_count=iter_count)
        write_image(grabcut_result, f"{image_names[i]}_grabcut_{iter_count}_iterations.png")

    print(" Graph Cuts has finished")
    """
    

print("Segmentation completed. Results saved in the THE3_Outputs folder.")