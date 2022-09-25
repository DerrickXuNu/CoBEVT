from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from opencood.data_utils.pre_processor.bev_preprocessor import BevPreprocessor
from opencood.data_utils.pre_processor.rgb_preprocessor import RgbPreProcessor


__all__ = {
    'BevPreprocessor': BevPreprocessor,
    'RgbPreprocessor': RgbPreProcessor
}


def build_preprocessor(preprocess_cfg, train):
    process_method_name = preprocess_cfg['core_method']
    error_message = f"{process_method_name} is not found. " \
                     f"Please add your processor file's name in opencood/" \
                     f"data_utils/processor/init.py"
    assert process_method_name in ['BasePreprocessor',
                                   'BevPreprocessor',
                                   'RgbPreprocessor'], \
        error_message

    processor = __all__[process_method_name](
        preprocess_params=preprocess_cfg,
        train=train
    )

    return processor
