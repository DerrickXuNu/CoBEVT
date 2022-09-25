"""
Post processing for rgb camera groundtruth
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor


class CameraBevPostprocessor(BasePostprocessor):
    """
    This postprocessor mainly transfer the uint bev maps to float.
    """

    def __init__(self, anchor_params, train):
        super(CameraBevPostprocessor, self).__init__(anchor_params, train)
        self.params = anchor_params
        self.train = train
        self.softmax = nn.Softmax(dim=1)

    def generate_label(self, bev_map):
        """
        Convert rgb images to binary output.

        Parameters
        ----------
        bev_map : np.ndarray
            Uint 8 image with 3 channels.
        """
        bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
        bev_map = np.array(bev_map, dtype=np.float) / 255.
        bev_map[bev_map > 0] = 1

        return bev_map

    def merge_label(self, road_map, lane_map):
        """
        Merge lane and road map into one.

        Parameters
        ----------
        static_map :
        lane_map :
        """
        merge_map = np.zeros((road_map.shape[0],
                              road_map.shape[1]))
        merge_map[road_map == 1] = 1
        merge_map[lane_map == 1] = 2

        return merge_map

    def softmax_argmax(self, seg_logits):
        output_prob = self.softmax(seg_logits)
        output_map = torch.argmax(output_prob, dim=1)

        return output_prob, output_map

    def post_process_train(self, output_dict):
        """
        Post process the output of bev map to segmentation mask.
        todo: currently only for single vehicle bev visualization.

        Parameters
        ----------
        output_dict : dict
            The output dictionary that contains the bev softmax.

        Returns
        -------
        The segmentation map. (B, C, H, W) and (B, H, W)
        """
        static_seg = output_dict['static_seg'][:, 0]
        dynamic_seg = output_dict['dynamic_seg'][:, 0]

        static_prob, static_map = self.softmax_argmax(static_seg)
        dynamic_prob, dynamic_map = self.softmax_argmax(dynamic_seg)

        output_dict.update({
            'static_prob': static_prob,
            'static_map': static_map,
            'dynamic_map': dynamic_map,
            'dynamic_prob': dynamic_prob
        })

        return output_dict

    def post_process(self, batch_dict, output_dict):
        # todo: rignt now we don't support late fusion (only no fusion)
        output_dict = self.post_process_train(output_dict)

        return output_dict
