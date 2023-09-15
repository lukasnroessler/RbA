import numpy as np
import open3d as o3d
import sys
import argparse
from numba import jit
from PIL import Image



parser = argparse.ArgumentParser(description='OOD Evaluation')

parser.add_argument('--eval_img', type=str, 
                    help= """Name the path to the evaluated image as .npy file""")

parser.add_argument('--depth_img', type=str, # action='store_true',
                    help="""Name path to depth image""")
parser.add_argument('--camera_fov', type=float,
                    help=""""Value for camera field of view""")
parser.add_argument('--voxel_size', type=float,
                    help=""""size for a single voxel""")


args = parser.parse_args()

def check_file_ending(path):
    file_ending = path.split(".")[-1]
    print("     ending", file_ending)
    if file_ending == "npy":
        return True
    return False





@jit(nopython=True)  # Compile the function with Numba
def calculate_carla_depth(depth_img):
    depth_img = Image.open(depth_img)

    # Get the dimensions of the image
    depth_img_arr = np.zeros((depth_img.height, depth_img.width,))
    # Loop through each pixel in the depth_img
    for row in range(depth_img.height):
        for col in range(depth_img.width):
            pixel = depth_img[row, col]
            depth_img_arr[row, col] = 1000 * (pixel[0] + pixel[1] * 256 + pixel[2] * 256 * 256) / (
                256 * 256 * 256 - 1
            )
    return depth_img_arr

def voxel_transform(scores, depths):
    height, width, _ = depths.shape

    camera_fov = args.camera_fov
    focal = width / (2.0 * np.tan(camera_fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    cx = width / 2.0  # from camera projection matrix
    cy = height / 2.0

    # voxel_size = args.voxel_size

    point_list = []
    anomaly_score_list = []

    for i in range(height):
        for j in range(width):
            z = depths[i,j]
            anomaly_score = scores[i][j]
            # depth encoded in rgb values. For further information take a look at carla docs
            # depth_color[2] = B
            # depth_color[1] = G
            # depth_color[0] = R

            x, y = (j - cx) * z / focal, (i - cy) * z / focal
            coordinate = [x,y,z]

            if np.linalg.norm(coordinate) > 100:
                continue
            else:
                point_list.append(coordinate)
                # red, green, blue = (
                #     semantic_color[0],
                #     semantic_color[1],
                #     semantic_color[2],
                # )
                # point_color = [
                #     i / 255.0 for i in [red, green, blue]
                # ]  # format to o3d color values
                anomaly_score_list.append(anomaly_score)

    depth_pcloud = o3d.geometry.PointCloud()  # create point cloud object
    depth_pcloud.points = o3d.utility.Vector3dVector(
        point_list
    )  # set pcd_np as the point cloud points
    # depth_pcloud.colors = o3d.utility.Vector3dVector(color_list)
    # point cloud needs to be rotated to fit with lidar point cloud in the future
    r_matrix = depth_pcloud.get_rotation_matrix_from_xyz(
        (-np.pi / 2, np.pi / 2, 0)
    )  # rotation matrix
    depth_pcloud.rotate(
        r_matrix, center=(0, 0, 0)
    )  # rotate depth point cloud to fit lidar point cloud
    return depth_pcloud    

    voxel_pcd = o3d.geometry.PointCloud()


def main():
    eval_image = np.load(args.eval_img)
    depth_img_arr = calculate_carla_depth(args.depth_img)



    return


if __name__ == "__main__":
    main()
    # for input in inputs:
    #     print(f"## Opening {input} ##")
    #     if check_file_ending(input):
    #         transform_npy(input)
    #     else:
    #         print("     --> wrong filetype")
