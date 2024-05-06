import os
import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


working_dir = os.path.join(os.getcwd(), '5sem/results/')


class FeaturesComputer:
    @classmethod
    def get_quarter_masses(cls, image):
        height, width = image.shape

        half_height = height // 2
        half_width = width // 2

        top_left = round(np.sum(image[:half_height, :half_width]), 3)
        top_right = round(np.sum(image[:half_height, half_width:]), 3)
        bottom_left = round(np.sum(image[half_height:, :half_width]), 3)
        bottom_right = round(np.sum(image[half_height:, half_width:]), 3)

        return (top_left, top_right, bottom_left, bottom_right)

    @classmethod
    def get_normalized_quarter_masses(cls, image):
        quarter_masses = cls.get_quarter_masses(image)
        height, width = image.shape
        quarter_area = (height // 2) * (width // 2)

        return tuple(round(k / quarter_area, 3) for k in quarter_masses)

    @classmethod
    def get_center_of_mass(cls, image):
        black_pixel_indices = np.argwhere(image == 1)
        
        center_y = round(np.mean(black_pixel_indices[:, 0]), 3)
        center_x = round(np.mean(black_pixel_indices[:, 1]), 3)
        
        return (center_x, center_y)

    @classmethod
    def get_normalized_center_of_mass(cls, image):
        center_x, center_y = cls.get_center_of_mass(image)
        height, width = image.shape
        
        normalized_center_x = round(center_x / width, 3)
        normalized_center_y = round(center_y / height, 3)
        
        return (normalized_center_x, normalized_center_y)

    @classmethod
    def get_axial_moments_of_inertia(cls, image):
        center_x, center_y = cls.get_center_of_mass(image)

        black_pixel_indices = np.argwhere(image == 1)

        horizontal_distances = black_pixel_indices[:, 1] - center_x
        vertical_distances = black_pixel_indices[:, 0] - center_y

        moment_x = round(np.sum(horizontal_distances ** 2), 3)
        moment_y = round(np.sum(vertical_distances ** 2), 3)

        return (moment_x, moment_y)

    @classmethod
    def get_normalized_axial_moments_of_inertia(cls, image):
        height, width = image.shape
        moment_x, moment_y = cls.get_axial_moments_of_inertia(image)

        normalized_moment_x = round(moment_x / (width ** 2 * height ** 2), 3)
        normalized_moment_y = round(moment_y / (height ** 2 * width ** 2), 3)

        return (normalized_moment_x, normalized_moment_y)

    @classmethod
    def get_x_y_profiles(cls, image):
        x_profile = np.sum(image, axis=0)
        y_profile = np.sum(image, axis=1)

        return (x_profile, y_profile)


def save_image_profiles_to_png(horizontal_profile, vertical_profile, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.bar(range(len(horizontal_profile)), horizontal_profile, color='blue')
    ax1.set_title('Горизонтальный профиль')
    ax1.set_xlabel('Столбцы')
    ax1.set_ylabel('Количество черных пикселей')

    ax2.bar(range(len(vertical_profile)), vertical_profile, color='orange')
    ax2.set_title('Вертикальный профиль')
    ax2.set_xlabel('Строки')
    ax2.set_ylabel('Количество черных пикселей')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    fig.tight_layout()

    plt.savefig(filename)
    plt.close(fig)


def handle_img(img_name):
    input_image = np.array(Image.open(working_dir + '/input/' + img_name).convert('L'))
    input_image = (input_image >= 128).astype(int)  # binarize image

    results = [
        FeaturesComputer.get_quarter_masses(input_image),
        FeaturesComputer.get_normalized_quarter_masses(input_image),
        FeaturesComputer.get_center_of_mass(input_image),
        FeaturesComputer.get_normalized_center_of_mass(input_image),
        FeaturesComputer.get_axial_moments_of_inertia(input_image),
        FeaturesComputer.get_normalized_axial_moments_of_inertia(input_image),
    ]

    # save to .csv
    with open(working_dir + '/output/result.csv', mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([img_name.split('.')[0], *results])

    horizontal_profile, vertical_profile = FeaturesComputer.get_x_y_profiles(input_image)

    save_image_profiles_to_png(
        horizontal_profile, 
        vertical_profile, 
        working_dir + '/output/' + img_name
    )

    
def main():
    for i in range(1, 2):
        handle_img(f'{i}.png')
        

if __name__ == "__main__":
    main()
