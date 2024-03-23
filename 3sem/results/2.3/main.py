import os

import numpy as np
from PIL import Image


working_dir = os.path.join(os.getcwd(), '3sem/results/2.3')


def xor_images(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape")

    xor_result = np.bitwise_xor(image1, image2)

    return xor_result


def handle_img(img_name):
    modes = ('default', 'oblique')

    for mode in modes:
        image1 = np.array(Image.open(working_dir + '/input/' + img_name).convert("L"))
        image2 = np.array(Image.open(working_dir + '/input/' + f'k=5_mode={mode}_{img_name}').convert("L"))
        xor_result = xor_images(image1, image2)
        Image.fromarray(xor_result.astype(np.uint8))\
            .save(working_dir + '/output/' + f'k=5_mode={mode}_{img_name}')


def main():
    images = tuple(f'{i}.png' for i in range(1, 4))

    for img in images:
        handle_img(img)   


if __name__ == "__main__":
    main()
