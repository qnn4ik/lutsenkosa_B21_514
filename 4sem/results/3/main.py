import os

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


working_dir = os.path.join(os.getcwd(), '4sem/results/3')


def _compute_otsu_criteria(im, th):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold
    # that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1


def otsu_threshold(img):
    threshold_range = range(int(np.max(img))+1)
    criterias = np.array([_compute_otsu_criteria(img, th) for th in threshold_range])

    # find threshold by minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]

    binary = img
    binary[binary > best_threshold] = 255
    binary[binary <= best_threshold] = 0

    return binary


def sobel_operator(image):
    sobel_x = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    sobel_y = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm')  # свертка
    gradient_x_norm = gradient_x * 255 / np.max(gradient_x)

    gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm')
    gradient_y_norm = gradient_y * 255 / np.max(gradient_y)

    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_norm = gradient * 255 / np.max(gradient)

    return (gradient_x_norm, gradient_y_norm, gradient_norm)


def handle_img(img_name):
    input_image = np.array(Image.open(working_dir + '/input/' + img_name).convert("L"))
    
    gradient_x_norm, gradient_y_norm, gradient_norm = sobel_operator(input_image)

    monochrome_image = otsu_threshold(gradient_norm)

    Image.fromarray(gradient_x_norm.astype(np.uint8))\
        .save(working_dir + '/output/' + f'Gx_{img_name}')

    Image.fromarray(gradient_y_norm.astype(np.uint8))\
        .save(working_dir + '/output/' + f'Gy_{img_name}')

    Image.fromarray(gradient_norm.astype(np.uint8))\
        .save(working_dir + '/output/' + f'G_{img_name}')
    
    Image.fromarray(monochrome_image.astype(np.uint8))\
        .save(working_dir + '/output/' + f'Gbin_{img_name}')


def main():
    images = tuple(f'{i}.png' for i in range(1, 5))

    for img in images[3:]:
        handle_img(img)   


if __name__ == "__main__":
    main()
