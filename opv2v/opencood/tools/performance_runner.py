"""
Performance runner to calculate model's params, flops and inference speed
"""
import argparse
import statistics
import time

import torch
from ptflops import get_model_complexity_info

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    test_tensor = torch.randn((1, 1, 4, 512, 512, 3))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        test_tensor = test_tensor.to(device)

    ave_time = []
    model.eval()

    macs, params = get_model_complexity_info(model, (1, 4, 512, 512, 3),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)

    print('Running inference time test')
    for i in range(100):
        with torch.no_grad():
            torch.cuda.synchronize()

            tsince = int(round(time.time() * 1000))
            output_dict = model({'inputs': test_tensor})
            time_elapsed = int(round(time.time() * 1000)) - tsince

            if i > 20:
                ave_time.append(time_elapsed)

    print('Average FPS: %f' % (1000/statistics.mean(ave_time)))


if __name__ == '__main__':
    main()
