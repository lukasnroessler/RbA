import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
import argparse
import os


parser = argparse.ArgumentParser(description='OOD Evaluation')



parser.add_argument('--dataset_path', type=str,
                    help=""""path to anovox root""")


args = parser.parse_args()


def get_voxel_grids():
    root = args.dataset_path
    voxel_data = []

    for scenario in os.listdir(root):
        if scenario == 'Scenario_Configuration_Files':
            continue
        voxel_dir = os.path.join(root, scenario, 'VOXEL_GRID')
        for grid in os.listdir(voxel_dir):
            voxel_data.append(os.path.join(voxel_dir, grid))        
    
    return sorted(voxel_data)




def evaluate_anomalies(pred, gt):
    pred = np.squeeze()
    gt = np.squeeze()
    anomaly_labels = [33,34]
    anomaly_mask = np.where(gt in anomaly_labels)

    

    pred_voxels = np.load(pred)
    gt_voxels = np.load(gt)

    voxel_labels = gt_voxels[:,:,:,]




def main():
    voxel_gts = get_voxel_grids()
    
    voxel_pred_dir = '/home/lukasnroessler/Projects/RbA/voxelpreds'
    # voxel_pred_dir = '/home/tes_unreal/Desktop/BA/RbA/voxelpreds'
    
    for i, voxel_pred in enumerate(os.listdir(voxel_pred_dir)):
        voxel_pred = os.path.join(voxel_pred_dir, voxel_pred)
        voxel_gt = voxel_gts[i]

        











if __name__ == "__main__":
    main()