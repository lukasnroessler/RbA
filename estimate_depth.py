import cv2

import torch

import numpy as np

import os

from PIL import Image

import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

import argparse

import re

parser = argparse.ArgumentParser(description='OOD Evaluation')



parser.add_argument('--dataset_path', type=str,
                    help=""""path to anovox root""")


output_path = "/home/lukasnroessler/Projects/RbA/VoxeldepthPreds"



args = parser.parse_args()

 
def collect_carla_img():
    root = args.dataset_path
    images = []

    for scenario in os.listdir(root):
        if scenario == 'Scenario_Configuration_Files':
            continue
        img_dir = os.path.join(root, scenario, 'RGB_IMG')
        for image in os.listdir(img_dir):
            images.append(os.path.join(img_dir, image))
    
    return sorted(images)



def depth_prediction(input_image_path):
    # Load a model (choose one from DPT_Large, DPT_Hybrid, MiDaS_small)

    model_type = "DPT_Large"  # Change this for different models

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    

    # Move model to GPU if available

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    midas.to(device)

    midas.eval()

    

    # Load transforms to resize and normalize the image for large or small model

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":

        transform = midas_transforms.dpt_transform

    else:

        transform = midas_transforms.small_transform

    

    # Load image and apply transforms

    # input_image_path = "Original.png"

    img = cv2.imread(input_image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    

    # Predict and resize to original resolution

    with torch.no_grad():

        prediction = midas(input_batch)

    

        prediction = torch.nn.functional.interpolate(

            prediction.unsqueeze(1),

            size=img.shape[:2],

            mode="bicubic",

            align_corners=False,

        ).squeeze()

    

    # output = prediction.cpu().numpy()
    return prediction.cpu().numpy()


# print(np.amax(output))
# print(np.amin(output))

# output = np.asarray(output)
# np.save("depth_map0.npy", output)
# output_img = Image.fromarray(output)
# output_img = output_img.convert("L")
# output_img.save("depth_map0.png")


# # Normalize depth map values to [0, 255]

# depth_map_normalized = (output - output.min()) / (output.max() - output.min()) * 255

# depth_map_normalized = depth_map_normalized.astype("uint8")

 

# # Save the depth map as "depth_map.jpg" in the same folder

# depth_map_image = ToPILImage()(depth_map_normalized)

# depth_map_image.save("depth_map.png")

 

# # Show the original image and depth map

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)

# plt.imshow(img)

# plt.title("Original Image")

# plt.subplot(1, 2, 2)

# plt.imshow(depth_map_normalized, cmap="jet")

# plt.title("Depth Map")

# plt.show()


def save_pil(img_name, img_arr):
    # Normalize depth map values to [0, 255]

    depth_map_normalized = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min()) * 255

    depth_map_normalized = depth_map_normalized.astype("uint8")

    

    # Save the depth map as "depth_map.jpg" in the same folder

    depth_map_image = ToPILImage()(depth_map_normalized)

    depth_map_image.save(img_name + '.png')



def get_img_identifier(img_name): # get id number of image

    img_name = img_name.replace('.png', '')
    identifier = re.findall(r'\d+', img_name)
    return identifier[-1]

def main():
    img_list = collect_carla_img()
    os.mkdir("/home/lukasnroessler/Projects/RbA/VoxeldepthPreds")


    for img in img_list:
        img_file = os.path.basename(img)
        img_id = get_img_identifier(img_file)
        
        pred = depth_prediction(img)


        new_file_name = 'depth_pred_' + img_id

        # pred = 1 / pred
        
        # pred = np.divide(1, pred, out=np.zeros_like(pred), where=pred!=0)
        # pred = np.ones_like(pred) - pred
        # save_pil(os.path.join(output_path, new_file_name), pred)


        depth_map_normalized = (pred - pred.min()) / (pred.max() - pred.min())

        # depth_map_normalized = np.ones_like(depth_map_normalized) - depth_map_normalized

        depth_map_normalized = 1 / depth_map_normalized

        depth_map_normalized = depth_map_normalized * 255

        depth_map_normalized = depth_map_normalized.astype("uint8")

        np.save(os.path.join(output_path, new_file_name), depth_map_normalized)

        





if __name__ == '__main__':
    main()
