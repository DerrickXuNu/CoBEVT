import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.tools.train_utils import save_bev_seg_binary, STD, MEAN


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def camera_inference_visualization(output_dict,
                                   batch_dict,
                                   output_dir,
                                   epoch,
                                   model_type='dynamic'):
    image_width = 800
    image_height = 600

    output_folder = os.path.join(output_dir, 'test_vis')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_images = \
        batch_dict['ego']['inputs'].detach().cpu().data.numpy()[0, 0]
    visualize_summary = np.zeros((image_height,
                                  image_width * 6,
                                  3),
                                 dtype=np.uint8)

    for j in range(raw_images.shape[0]):
        raw_image = 255 * ((raw_images[j] * STD) + MEAN)
        raw_image = np.array(raw_image, dtype=np.uint8)
        # rgb = bgr
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, (image_width, image_height))

        visualize_summary[:, image_width * j:image_width * (j + 1)] = raw_image

    if model_type == 'dynamic':
        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[0,
                                                                        0]
        gt_dynamic = np.array(gt_dynamic * 255., dtype=np.uint8)
        gt_dynamic = cv2.resize(gt_dynamic, (image_width,
                                             image_height))
        gt_dynamic = cv2.cvtColor(gt_dynamic, cv2.COLOR_GRAY2BGR)

        pred_dynamic = \
            output_dict['dynamic_map'].detach().cpu().data.numpy()[0]
        pred_dynamic = np.array(pred_dynamic * 255., dtype=np.uint8)
        pred_dynamic = cv2.resize(pred_dynamic, (image_width,
                                                 image_height))
        pred_dynamic = cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)
        visualize_summary[:, image_width * 4:image_width * 5] = gt_dynamic
        visualize_summary[:, image_width * 5:] = pred_dynamic

    else:
        gt_static_origin = \
            batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[0, 0]
        gt_static = np.zeros((gt_static_origin.shape[0],
                              gt_static_origin.shape[1],
                              3), dtype=np.uint8)
        gt_static[gt_static_origin == 1] = np.array([88, 128, 255])
        gt_static[gt_static_origin == 2] = np.array([244, 148, 0])

        pred_static_origin = \
            output_dict['static_map'].detach().cpu().data.numpy()[0]
        pred_static = np.zeros((pred_static_origin.shape[0],
                                pred_static_origin.shape[1],
                                3), dtype=np.uint8)
        pred_static[pred_static_origin == 1] = np.array([88, 128, 255])
        pred_static[pred_static_origin == 2] = np.array([244, 148, 0])

        gt_static = cv2.resize(gt_static, (image_width,
                                           image_height))
        pred_static = cv2.resize(pred_static, (image_width,
                                               image_height))

        visualize_summary[:, image_width * 4:image_width * 5] = gt_static
        visualize_summary[:, image_width * 5:] = pred_static

    cv2.imwrite(os.path.join(output_folder, '%04d.png')
                % epoch, visualize_summary)
