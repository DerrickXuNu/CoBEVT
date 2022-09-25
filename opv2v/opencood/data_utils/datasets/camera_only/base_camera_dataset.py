"""
A plain dataset class for cameras
"""
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.data_utils.datasets
from opencood.utils import box_utils, common_utils, camera_utils
from opencood.data_utils.datasets import basedataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor


class BaseCameraDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(BaseCameraDataset, self).__init__(params, visualize, train,
                                                validate)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def get_sample_random(self, idx):
        base_data_dict = self.retrieve_base_data(idx, True)

        return self.get_data_sample(base_data_dict)

    def get_sample(self, scenario_idx, timestamp_index):
        """
        Get data sample from scenario index and timestamp index directly.
        """
        base_data_dict = \
            self.retrieve_base_data((scenario_idx, timestamp_index),
                                    True)
        return self.get_data_sample(base_data_dict)

    def get_data_sample(self, base_data_dict):
        processed_data_dict = OrderedDict()

        ego_id, ego_lidar_pose = self.find_ego_pose(base_data_dict)

        # used to save all object coordinates under ego space
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = common_utils.cav_distance_cal(selected_cav_base,
                                                     ego_lidar_pose)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue
            processed_data_dict[cav_id] = base_data_dict[cav_id]
            # the objects bbx position under ego and cav lidar coordinate frame
            object_bbx_ego, object_bbx_cav, object_ids = \
                self.get_item_single_car(selected_cav_base,
                                         ego_lidar_pose)

            object_stack.append(object_bbx_ego)
            object_id_stack += object_ids

            processed_data_dict[cav_id]['object_bbx_cav'] = object_bbx_cav
            processed_data_dict[cav_id]['object_id'] = object_ids

        # Object stack contains all objects that can be detected from all
        # cavs nearby under ego coordinates. We need to exclude the repititions
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((100, 7))
        mask = np.zeros(100)
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # update the ego vehicle with all objects coordinates
        processed_data_dict[ego_id]['object_bbx_ego'] = object_bbx_center
        processed_data_dict[ego_id]['object_bbx_ego_mask'] = mask

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Get the selected vehicle's camera
        Parameters
        ----------
        selected_cav_base : dict
            The basic information of the selected vehicle.

        ego_pose : list
            The ego vehicle's (lidar) pose.

        Returns
        -------
        objects coordinates under ego coordinate frame and corresponding
        object ids.
        """

        # generate the bounding box(n, 7) under the ego space
        object_bbx_center_ego, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)
        # generate the bounding box under the cav space
        object_bbx_center_cav, object_bbx_mask_cav, _ = \
            self.post_processor.generate_object_center(
                [selected_cav_base],
                selected_cav_base['params']['lidar_pose'])

        return object_bbx_center_ego[object_bbx_mask == 1], \
               object_bbx_center_cav[object_bbx_mask_cav == 1], \
               object_ids

    def visualize_agent_camera_bbx(self, agent_sample,
                                   camera='camera0', draw_3d=True,
                                   color=(0, 255, 0), thickness=2):
        """
        Visualize bbx on the 2d image for a certain agent
        and a certain camera.

        Parameters
        ----------
        agent_sample : dict
            The dictionary contains a certain agent information at a certain
            timestamp.

        camera : str
            Which camera to visualize bbx.

        draw_3d : bool
            Draw 2d bbx or 3d bbx on image.

        color : tuple
            Bbx draw color.

        thickness : int
            Draw thickness.

        Returns
        -------
        The drawn image.
        """
        assert camera in ['camera0', 'camera1', 'camera2', 'camera3'], \
            'the camera has to be camera0, camera1, camera2 or camera3'

        # load camera params and rgb image
        camera_rgb = agent_sample['camera_np'][camera]
        camera_param = agent_sample['camera_params'][camera]
        camera_extrinsic = camera_param['camera_extrinsic']
        camera_intrinsic = camera_param['camera_intrinsic']

        # objects coordinate
        objects = agent_sample['object_bbx_cav']
        # convert to corner representation
        objects = box_utils.boxes_to_corners_3d(objects,
                                                self.post_processor.params[
                                                    'order'])
        # project objects coordinate from lidar space to camera space
        object_camera = camera_utils.project_3d_to_camera(objects,
                                                          camera_intrinsic,
                                                          camera_extrinsic)
        if draw_3d:
            draw_rgb = camera_utils.draw_3d_bbx(camera_rgb,
                                                object_camera,
                                                color,
                                                thickness)
        else:
            draw_rgb = camera_utils.draw_2d_bbx(camera_rgb,
                                                objects,
                                                color,
                                                thickness)
        return draw_rgb

    def visualize_agent_bbx(self, data_sample, agent, draw_3d=True,
                            color=(0, 255, 0), thickness=2):
        """
        Draw bbx on a certain agent's all cameras.

        Parameters
        ----------
        data_sample : dict
            The sample contains all information of all agents.

        agent : str
            The target agent.

        draw_3d : bool
            Draw 3d or 2d bbx.

        color : tuple
            Bbx draw color.

        thickness : int
            Draw thickness.

        Returns
        -------
        A list of drawn image.
        """
        agent_sample = data_sample[agent]
        draw_image_list = []

        for camera in ['camera0', 'camera1', 'camera2', 'camera3']:
            draw_image = self.visualize_agent_camera_bbx(agent_sample,
                                                         camera,
                                                         draw_3d,
                                                         color,
                                                         thickness)
            draw_image_list.append(draw_image)

        return draw_image_list

    def visualize_all_agents_bbx(self, data_sample,
                                 draw_3d=True,
                                 color=(0, 255, 0),
                                 thickness=2):
        """
        Visualize all agents and all cameras in a certain frame.
        """
        draw_image_list = []
        cav_id_list = []

        for cav_id, cav_content in data_sample.items():
            draw_image_list.append(self.visualize_agent_bbx(data_sample,
                                                            cav_id,
                                                            draw_3d,
                                                            color,
                                                            thickness))
            cav_id_list.append(cav_id)

        return draw_image_list, cav_id_list


if __name__ == '__main__':
    params = load_yaml('../../../hypes_yaml/opcamera/base_camera.yaml')

    opencda_dataset = BaseCameraDataset(params, train=True, visualize=True)
    data_example = opencda_dataset.get_sample(4, 10)
    draw_image_list, cav_id_list =\
        opencda_dataset.visualize_all_agents_bbx(data_example)

    camera_utils.plot_all_agents(draw_image_list, cav_id_list)
