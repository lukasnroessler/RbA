import imageio.v3 as iio
import numpy as np
import sys


def image_properties(inputs):
    for input in inputs:
        image = np.load(input)
        print("shape", image.shape)
        print("max", np.amax(image))
        print("min", np.amin(image))

if __name__ == "__main__":
    inputs = sys.argv[1:]
    image_properties(inputs)



