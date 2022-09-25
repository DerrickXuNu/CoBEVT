import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Seed Setting----------------------')
    seed = train_utils.init_random_seed(None if opt.seed == 0 else opt.seed)
    hypes['train_params']['seed'] = seed
    print('Set seed to %d' % seed)
    train_utils.set_random_seed(seed)

    print('-----------------Dataset Building------------------')

    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_val_dataset = build_dataset(hypes, visualize=False, train=True,
                                         validate=True)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_val_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch)
        val_loader = DataLoader(opencood_val_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_val_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)


    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # lr scheduler setup
    epoches = hypes['train_params']['epoches']
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    print('Training start with num steps of %d' % num_steps)
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):

        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego'])

            criterion.logging(epoch, i, len(train_loader), writer,
                              pbar=pbar2)
            pbar2.update(1)

            # update the lr to tensorboard
            for lr_idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar('lr_%d' % lr_idx, param_group["lr"],
                                  epoch * num_steps + i)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            dynamic_ave_iou = []
            static_ave_iou = []
            lane_ave_iou = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])

                    final_loss = criterion(output_dict,
                                           batch_data['ego'])
                    valid_ave_loss.append(final_loss.item())

                    # visualization purpose
                    output_dict = \
                        opencood_val_dataset.post_process(batch_data['ego'],
                                                          output_dict)
                    train_utils.save_bev_seg_binary(output_dict,
                                                    batch_data,
                                                    saved_path,
                                                    i,
                                                    epoch)
                    iou_dynamic, iou_static = cal_iou_training(batch_data,
                                                               output_dict)
                    static_ave_iou.append(iou_static[1])
                    dynamic_ave_iou.append(iou_dynamic[1])
                    lane_ave_iou.append(iou_static[2])

            valid_ave_loss = statistics.mean(valid_ave_loss)
            static_ave_iou = statistics.mean(static_ave_iou)
            lane_ave_iou = statistics.mean(lane_ave_iou)
            dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

            print('At epoch %d, the validation loss is %f,'
                  'the dynamic iou is %f, t'
                  'he road iou is %f'
                  'the lane ious is %f' % (epoch,
                                           valid_ave_loss,
                                           dynamic_ave_iou,
                                           static_ave_iou,
                                           lane_ave_iou))

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            writer.add_scalar('Dynamic_Iou', dynamic_ave_iou, epoch)
            writer.add_scalar('Road_IoU', static_ave_iou, epoch)
            writer.add_scalar('Lane_IoU', static_ave_iou, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        opencood_train_dataset.reinitialize()


if __name__ == '__main__':
    main()
