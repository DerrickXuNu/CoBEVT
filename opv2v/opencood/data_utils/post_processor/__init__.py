from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.data_utils.post_processor.lidar_bev_postprocessor import LidarBevPostprocessor
from opencood.data_utils.post_processor.camera_bev_postprocessor import CameraBevPostprocessor

__all__ = {
    'BevPostprocessor': LidarBevPostprocessor,
    'BasePostprocessor': BasePostprocessor,
    'CameraBevPostprocessor': CameraBevPostprocessor
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor',
                                   'BevPostprocessor',
                                   'BasePostprocessor',
                                   'CameraBevPostprocessor']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        train=train
    )

    return anchor_generator
