import cv2

import torch

import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

 

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

input_image_path = "Original.png"

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

 

output = prediction.cpu().numpy()

 

# Normalize depth map values to [0, 255]

depth_map_normalized = (output - output.min()) / (output.max() - output.min()) * 255

depth_map_normalized = depth_map_normalized.astype("uint8")

 

# Save the depth map as "depth_map.jpg" in the same folder

depth_map_image = ToPILImage()(depth_map_normalized)

depth_map_image.save("depth_map.png")

 

# Show the original image and depth map

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.imshow(img)

plt.title("Original Image")

plt.subplot(1, 2, 2)

plt.imshow(depth_map_normalized, cmap="jet")

plt.title("Depth Map")

plt.show()