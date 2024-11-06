import numpy as np
import cv2 
import matplotlib.pyplot as  plt

input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'


def read_image(filename, gray_scale=False):
    # Read the image in grayscale or color
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE)
        return img
    img = cv2.imread(input_folder + filename)
    return img

def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    cv2.imwrite(output_folder+filename, img)



def rotate_upsample(img, scale, degree, interpolation_type):
    '''img: img to be rotated and upsampled
    scale: scale of upsampling (e.g. if current width and height is 64x64, and scale is 4, wodth and height of the output should be 256x256)
    degree: shows the degree of rotation
    interp: either linear or cubic'''

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if interpolation_type == 'linear':
        interp = cv2.INTER_LINEAR
    elif interpolation_type == 'cubic':
        interp = cv2.INTER_CUBIC
    else:
        raise ValueError("Interpolation type must be 'linear' or 'cubic'")


    (h, w) = img.shape[:2]  

    upsampled_img = cv2.resize(img,None, fx= scale,fy=scale, interpolation=interp)

    center = (int(w * scale) // 2 , int(h * scale) // 2 )  

    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=-1*degree, scale=1.0)

    corrected_image = cv2.warpAffine(upsampled_img, rotation_matrix, (int(w * scale), int(h * scale)))

    return corrected_image






def cropping_way(img1, img2):

    img1_rows, img1_columns = img1.shape[:2]
    img2_rows, img2_columns = img2.shape[:2]

    if img1_rows > img2_rows or img1_columns > img2_columns:
        y_start = (img1_rows - img2_rows) // 2
        x_start = (img1_columns - img2_columns) // 2

        img1_cropped = img1[y_start:y_start + img2_rows, x_start:x_start + img2_columns]
        return img1_cropped,img2
    else:
        y_start = (img2_rows - img1_rows) // 2
        x_start = (img2_columns - img1_columns) // 2

        img2_cropped = img2[y_start:y_start + img1_rows, x_start:x_start + img1_columns]
        return img1,img2_cropped


    


def padding_way(img1, img2):

    img1_rows, img1_columns = img1.shape[:2]
    img2_rows, img2_columns = img2.shape[:2]
    max_h = max(img1_rows, img2_rows)
    max_w = max(img1_columns, img2_columns)

    padded_img1 = np.zeros((max_h, max_w, 3), dtype=img1.dtype)
    padded_img2 = np.zeros((max_h, max_w, 3), dtype=img2.dtype)

    start_y1 = (max_h - img1_rows) // 2
    start_x1 = (max_w - img1_columns) // 2
    start_y2 = (max_h - img2_rows) // 2
    start_x2 = (max_w - img2_columns) // 2

    padded_img1[start_y1:start_y1 + img1_rows, start_x1:start_x1 + img1_columns] = img1
    padded_img2[start_y2:start_y2 + img2_rows, start_x2:start_x2 + img2_columns] = img2
    return padded_img1, padded_img2




def compute_distance(img1, img2):

    if img1.shape != img2.shape:
        #img1_new,img2_new = cropping_way(img1, img2)   
        img1_new,img2_new = padding_way(img1, img2)   # better
    else:
        img1_new,img2_new = img1,img2
        
    mse = np.mean((img1_new.astype(np.float32) - img2_new.astype(np.float32)) ** 2)
    return mse




def rgb_to_hsi(image):
 
    rgb = image.astype('float') / 255.0
    b, g, r = cv2.split(rgb)

  
    I = (r + g + b) / 3.0

    numerator = (r - g) + (r - b)
    denominator = 2 * np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-10 
    theta = np.arccos(numerator / denominator)
    H = np.zeros_like(I)

    H[b <= g] = theta[b <= g]
    H[g < b] = 2 * np.pi - theta[g < b]

    S = 1 - 3 * (np.minimum(r,np.minimum(g,b))) / (I+1e-10)


    hsi = cv2.merge([H, S, I])
    return hsi


def compute_hue_histogram(hsi, bins=256):   
    """Compute the histogram of the hue channel."""
    hue_channel = hsi[:, :, 0]
    hist, _ = np.histogram(hue_channel, bins=bins, range=(0, 2 * np.pi))
    hist = hist.astype('float')
    hist /= hist.sum()   
    return hist




def kl_divergence(p, q):

    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    q = np.clip(q, 1e-10, None)

    divergence = np.sum(np.where(p != 0, p * np.log(p / q+1e-10), 0))

    return divergence if not np.isnan(divergence) else 0






def desert_or_forest(img):
    '''img: image to be classified as desert or forest
    return a string: either 'desert'  or 'forest' 
    
    You should compare the KL Divergence between histograms of hue channel. 
    Please provide images and discuss these histograms in your report'''


    desert1 = read_image('desert1.jpg')
    desert2 = read_image('desert2.jpg')
    forest1 = read_image('forest1.jpg')
    forest2 = read_image('forest2.jpg')

    hsi_img = rgb_to_hsi(img)
    hsi_desert1 = rgb_to_hsi(desert1)
    hsi_desert2 = rgb_to_hsi(desert2)
    hsi_forest1 = rgb_to_hsi(forest1)
    hsi_forest2 = rgb_to_hsi(forest2)

    hist_img = compute_hue_histogram(hsi_img)
    hist_desert1 = compute_hue_histogram(hsi_desert1)
    hist_desert2 = compute_hue_histogram(hsi_desert2)
    hist_forest1 = compute_hue_histogram(hsi_forest1)
    hist_forest2 = compute_hue_histogram(hsi_forest2)


    kl_desert1 = kl_divergence(hist_img, hist_desert1)
    kl_desert2 = kl_divergence(hist_img, hist_desert2)
    kl_forest1 = kl_divergence(hist_img, hist_forest1)
    kl_forest2 = kl_divergence(hist_img, hist_forest2)

   
    desert_kl_min = min(kl_desert1, kl_desert2)
    forest_kl_min = min(kl_forest1, kl_forest2)

    if desert_kl_min < forest_kl_min:
        return 'desert'
    else:
        return 'forest'






def difference_images_gray(img1, img2, threshold=75):
    '''img1 and img2 are the images to take the difference from.
    returns the masked image where the object is highlighted.'''

    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    diff = cv2.absdiff(img1_gray, img2_gray)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    masked_image = cv2.bitwise_and(img2, img2, mask=mask.astype(np.uint8))

    return masked_image


    


def difference_images_rgb(img1, img2, threshold= 50):
   
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)
    
    diff_r = cv2.absdiff(r2, r1)
    diff_g = cv2.absdiff(g2, g1)
    diff_b = cv2.absdiff(b2, b1)
    
    diff_weighted = cv2.addWeighted(diff_r, 0.5, diff_g, 0.25, 0)
    diff_weighted = cv2.addWeighted(diff_weighted, 1.0, diff_b, 0.25, 0)
    
    _, mask = cv2.threshold(diff_weighted, threshold , 255, cv2.THRESH_BINARY)

    kernel = np.ones((11,11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_and(img2, img2, mask=mask)
    
    return result




if __name__ == '__main__':
    
    ###################### Q1
    # Read original image
    img_original = read_image('q1_1.png')
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    # Read corrupted image
    img = read_image('ratio_4_degree_30.png')
    # Correct the image with linear interpolation
    corrected_img_linear = rotate_upsample(img, 4, 30, 'linear')
    write_image(corrected_img_linear, 'q1_1_corrected_linear.png')
    # Correct the image with cubic interpolation
    corrected_img_cubic = rotate_upsample(img, 4, 30, 'cubic')
    write_image(corrected_img_cubic, 'q1_1_corrected_cubic.png')

    # Report the distances
    print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

    # Repeat the same steps for the second image
    img_original = read_image('q1_2.png')
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img = read_image('ratio_8_degree_45.png')
    corrected_img_linear = rotate_upsample(img, 8, 45, 'linear')
    write_image(corrected_img_linear, 'q1_2_corrected_linear.png')
    corrected_img_cubic = rotate_upsample(img, 8, 45, 'cubic')
    write_image(corrected_img_cubic, 'q1_2_corrected_cubic.png')

    # Report the distances
    print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

    ###################### Q2
    img = read_image('q2_1.jpg')
    result = desert_or_forest(img)
    print("Given image q2_1 is an image of a ", result)

    img = read_image('q2_2.jpg')
    result = desert_or_forest(img)
    print("Given image q2_2 is an image of a ", result)

    ###################### Q3
    img1 = read_image('q3_a1.png',gray_scale=True)
    img2 = read_image('q3_a2.png',gray_scale=True)
    result = difference_images_gray(img1,img2)
    write_image(result, 'masked_image_a.png')

    img1 = read_image('q3_b1.png')
    img2 = read_image('q3_b2.png')
    result = difference_images_rgb(img1,img2)
    write_image(result, 'masked_image_b.png')



