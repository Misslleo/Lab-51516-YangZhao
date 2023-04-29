'''Yang Create'''
from model import *
from local_occ_grid_map import LocalMap
import torch
from MaptoImage import train_dataset, train_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
P_prior = 0.5  # Prior occupancy probability
P_occ = 0.7  # Probability that cell is occupied with total confidence
P_free = 0.3  # Probability that cell is free with total confidence
MAP_X_LIMIT = [0, 6.4]  # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]  # Map limits on the y-axis
RESOLUTION = 0.1  # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8  # Occupancy threshold
counter = 0
BATCH_SIZE = 128
SEQ_LEN = 10
IMG_SIZE = 64
SPACE = " "

val_dataset = VaeTestDataset('C:/Users/Yang Zhao/Desktop/2023/Graph review paper/OGM-datasets/OGM-Turtlebot2/SGAN_val', "val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, \
                                               shuffle=False, drop_last=True)

num_batches = int(len(val_dataset) / val_dataloader.batch_size)

device = torch.device("cuda", 0)

with torch.no_grad():
    for i, batch in tqdm(enumerate(val_dataloader), total=num_batches):
        # for i, batch in enumerate(dataloader, 0):
        counter += 1
        # collect the samples as a batch:
        scans = batch['scan']
        scans = scans.to(device)
        positions = batch['position']
        positions = positions.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)
        # create occupancy maps:
        batch_size = scans.size(0)
        #create mask grid maps
        mask_gridMap = LocalMap(X_lim=MAP_X_LIMIT,
                                Y_lim=MAP_Y_LIMIT,
                                resolution=RESOLUTION,
                                p=P_prior,
                                size=[batch_size, SEQ_LEN],
                                device=device)
        # robot positions:
        x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        # Lidar measurements:
        distances = scans[:, SEQ_LEN:]
        # the angles of lidar scan: -135 ~ 135 degree
        angles = torch.linspace(-(135 * np.pi / 180), 135 * np.pi / 180, distances.shape[-1]).to(device)
        # Lidar measurement in X-Y plane：
        distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
        mask_binary_maps = mask_binary_maps.unsqueeze(2)

        fig = plt.figure(figsize=(8, 1))
        for m in range(SEQ_LEN):
            # display the mask of occupancy grids:
            a = fig.add_subplot(1, 10, m + 1)
            mask = mask_binary_maps[0, m]
            input_grid = make_grid(mask.detach().cpu())
            input_image = input_grid.permute(1, 2, 0)
            plt.imshow(input_image)
            plt.xticks([])
            plt.yticks([])
            fontsize = 8
            #input_title = "n=" + str(m + 1)
            #a.set_title(input_title, fontdict={'fontsize': fontsize})
        input_img_name = "C:/Users/Yang Zhao/PycharmProjects/rob/outputt/val/mask" + str(i) + ".jpg"
        plt.savefig(input_img_name)
        #plt.show()

        print(i)