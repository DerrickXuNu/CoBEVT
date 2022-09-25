from pathlib import Path
from tqdm import tqdm

import torch
import statistics
import time
import pytorch_lightning as pl
import hydra

from cross_view_transformer.common import setup_config, setup_network, setup_data_module


def setup(cfg):
    print('Benchmark mixed precision by adding +mixed_precision=True')
    print('Benchmark cpu performance +device=cpu')

    cfg.loader.batch_size = 1

    if 'mixed_precision' not in cfg:
        cfg.mixed_precision = False

    if 'device' not in cfg:
        cfg.device = 'cuda'


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    pl.seed_everything(2022, workers=True)

    network = setup_network(cfg)
    data = setup_data_module(cfg)
    loader = data.train_dataloader(shuffle=False)

    device = torch.device(cfg.device)
    print(device)

    network = network.to(device)
    network.eval()

    sample = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
    time_list = []
    with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
        with torch.no_grad():
            for _ in tqdm(range(1000+1)):
                torch.cuda.synchronize()
                start_time = time.time()
                network(batch)
                torch.cuda.synchronize()
                duration = time.time()-start_time
                time_list.append(duration)

    print('average time: %f ms' % statistics.mean(time_list))


if __name__ == '__main__':
    main()
