import imageio.v3 as iio
from PIL import Image
import numpy as np
import sys


def image_properties(inputs):
    for input in inputs:
        image = Image.open(input)
        image = np.asarray(image)
        # image = image[:,:,-1:]
        print("shape", image.shape)
        print("max", np.amax(image))
        print("min", np.amin(image))
        # np.squeeze(image, axis=2)
        image = Image.fromarray(image)
        # image.show()


def cut(image):
    # image = Image.open(input_path)
    image_arr = np.asarray(image)
    image_arr = np.copy(image_arr)[:,:,-1:]
    print(image_arr.shape)
    # image = Image.fromarray(image_arr)
    # ("/home/tes_unreal/Desktop/BA/RbA/cutimage.png", image_arr)
    # image.show()
    return image_arr

def mask(input_path):
    input = iio.imread(input_path)
    copy = np.copy(input)
    sliced_input = cut(copy)
    mask = np.where(sliced_input == 14)
    copy[mask] = 255
    iio.imwrite('/home/tes_unreal/Desktop/BA/RbA/mask.png', copy)

if __name__ == "__main__":
    inputs = sys.argv[1:]
    image_properties(inputs)
    # cut(Image.open(inputs[0]))
    mask(inputs[0])

    



