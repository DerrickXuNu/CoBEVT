"""
Merge static and dynamic gt and pred together
"""

import argparse
import cv2
import os
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dynamic_path', type=str, required=True,
                        help='Where is the dynamic inference savings,'
                             'e.g., logs/corpbevt')
    parser.add_argument('--static_path', type=str,
                        help='Where is the static inference savings')
    parser.add_argument('--output_path', type=str,
                        help='Where is the output dir')
    opt = parser.parse_args()
    return opt


def main():
    opt = arg_parser()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    dynamic_path = os.path.join(opt.dynamic_path, 'test_vis')
    static_path = os.path.join(opt.static_path, 'test_vis')

    dynamic_figures = os.listdir(dynamic_path)
    static_figures = os.listdir(static_path)

    image_width = 800
    image_height = 600

    assert len(dynamic_figures) == len(static_figures)

    for figure in dynamic_figures:
        dynamic_fig = cv2.imread(os.path.join(dynamic_path, figure), 0)
        static_fig = cv2.imread(os.path.join(static_path, figure))

        dynamic_gt = dynamic_fig[:, 4 * image_width:5 * image_width]
        dynamic_gt[dynamic_gt > 0] = 1
        dynamic_pred = dynamic_fig[:, 5 * image_width:]
        dynamic_pred[dynamic_pred > 0] = 1

        static_gt = static_fig[:, 4 * image_width:5 * image_width]
        static_pred = static_fig[:, 5 * image_width:]

        static_gt[dynamic_gt == 1] = np.array([255, 255, 255])
        static_pred[dynamic_pred == 1] = np.array([255, 255, 255])

        static_fig[:, 4 * image_width:5 * image_width] = static_gt
        static_fig[:, 5 * image_width:] = static_pred
        cv2.imwrite(os.path.join(opt.output_path, figure), static_fig)


if __name__ =='__main__':
    main()