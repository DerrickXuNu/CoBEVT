import os
import argparse
import opencood.data_utils.datasets
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils, common_utils, camera_utils
from opencood.data_utils.datasets.camera_only.base_camera_dataset import BaseCameraDataset

def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--scene', type=int, default=4,
                        help='The ith scene to visiualize')
    parser.add_argument('--sample', type=int, default=10,
                        help='The jth sample in the scene')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/opcamera/base_camera.yaml'))

    opencda_dataset = BaseCameraDataset(params, train=True, visualize=True)
    opt = vis_parser()

    data_example = opencda_dataset.get_sample(opt.scene, opt.sample)
    draw_image_list, cav_id_list =\
        opencda_dataset.visualize_all_agents_bbx(data_example)

    camera_utils.plot_all_agents(draw_image_list, cav_id_list)
