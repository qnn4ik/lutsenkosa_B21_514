import os
from typing import Literal

import numpy as np
from PIL import Image


working_dir = os.path.join(os.getcwd(), '3sem/results/1.3')


def median_filter(image, kernel_size, mode: Literal['default', 'oblique']='default'):
    """
    Meadian filter
    replaces each pixel's value with the median value of its neighborhood
    """
    # handle borders
    padded_image = np.pad(
        image, 
        ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
        mode='constant'
    )

    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):    
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]

            if mode == 'oblique':
                neighborhood = np.zeros(kernel_size)

                for k in range(kernel_size):  # oblique (косой) cross
                    row = i - kernel_size // 2 + k
                    col = j + kernel_size // 2 - k
                    
                    if row >= 0 and row < image.shape[0] and col >= 0 and col < image.shape[1]:
                        neighborhood[k] = padded_image[row, col]

            filtered_image[i, j] = np.median(neighborhood)

    return filtered_image


def handle_img(img_name):
    kernel_size = 7
    modes = ('default', 'oblique')

    monochrome_image_path = working_dir + '/input/' + img_name
    monochrome_image = np.array(Image.open(monochrome_image_path).convert("L"))

    for mode in modes:
        if mode == 'default':
            filtered_image = median_filter(monochrome_image, kernel_size)
            filtered_image_path = working_dir + '/output/' + f'k={kernel_size}_mode={mode}_{img_name}'
            Image.fromarray(filtered_image).save(filtered_image_path)
        elif mode == 'oblique':
            filtered_image = median_filter(monochrome_image, kernel_size, mode='oblique')
            filtered_image_path = working_dir + '/output/' + f'k={kernel_size}_mode={mode}_{img_name}'
            Image.fromarray(filtered_image).save(filtered_image_path)


def main():
    images = tuple(f'{i}.png' for i in range(1, 4))

    for img in images:
        handle_img(img)   


if __name__ == "__main__":
    main()
