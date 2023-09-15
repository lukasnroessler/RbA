import numpy as np
import open3d as o3d
import sys

def check_file_ending(path):
    file_ending = path.split(".")[-1]
    print("     ending", file_ending)
    if file_ending == "npy":
        return True
    return False


def transform_npy(file_path):
    try:
        img_arr = np.load(file_path)
        
    except Exception as e:
        print("Could not open file:", file_path)
        print(e)

if __name__ == "__main__":
    inputs = sys.argv[1:]
    for input in inputs:
        print(f"## Opening {input} ##")
        if check_file_ending(input):
            transform_npy(input)
        else:
            print("     --> wrong filetype")