import os

import numpy as np
import open3d as o3d

from opencood.visualization.vis_utils import visualize_sequence_sample_output, \
    visualize_single_sample_output_gt

if __name__ == "__main__":
    save_npy_path = '../logs/' \
                    'point_pillar_late_fusion_low_res_2021_09_08_09_55_22/npy'

    pred_list = sorted([os.path.join(save_npy_path, x)
                        for x in os.listdir(save_npy_path) if 'pred' in x])
    gt_list = sorted([os.path.join(save_npy_path, x)
                      for x in os.listdir(save_npy_path) if 'gt' in x])
    pcd_list = sorted([os.path.join(save_npy_path, x)
                       for x in os.listdir(save_npy_path) if 'pcd' in x])

    pred_tensor_list = [np.load(x) for x in pred_list]
    gt_tensor_list = [np.load(x) for x in gt_list]
    pcd_tensor_list = [np.load(x) for x in pcd_list]

    visualize_sequence_sample_output(pred_tensor_list,
                                     gt_tensor_list,
                                     pcd_tensor_list)
