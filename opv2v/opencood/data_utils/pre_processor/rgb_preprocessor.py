"""
Pre-processing for RGB cameras
"""

import cv2
import numpy as np

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class RgbPreProcessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(RgbPreProcessor, self).__init__(preprocess_params, train)

    def preprocess(self, rgb_image):
        rgb_image = self.channel_swap(rgb_image)
        rgb_image = self.resize_image(rgb_image)
        rgb_image = self.normalize(rgb_image)
        rgb_image = self.standalize(rgb_image)

        return rgb_image

    def standalize(self, rgb_image):
        mean = np.array(self.params['args']['mean'])
        std = np.array(self.params['args']['std'])

        rgb_image = (rgb_image - mean) / std

        return rgb_image

    def normalize(self, rgb_image):
        return np.array(rgb_image, dtype=float) / 255.

    def channel_swap(self, rgb_image):
        """
        Convert BGR to RGB if needed
        """
        if self.params['args']['bgr2rgb']:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = rgb_image

        return rgb_image

    def resize_image(self, rgb_image):
        """
        Resize image to the correct resolution.
        """
        resize_x = self.params['args']['resize_x']
        resize_y = self.params['args']['resize_y']

        rgb_image = cv2.resize(rgb_image, (resize_x, resize_y))

        return rgb_image
