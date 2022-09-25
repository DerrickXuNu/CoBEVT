import torch
import torch.nn as nn

from einops import rearrange


class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super(VanillaSegLoss, self).__init__()

        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        self.d_coe = args['d_coe']
        self.s_coe = args['s_coe']
        self.target = args['target']

        self.loss_func_static = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_dynamic = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.d_weights]).cuda())

        self.loss_dict = {}

    def forward(self, output_dict, gt_dict):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """

        static_pred = output_dict['static_seg']
        dynamic_pred = output_dict['dynamic_seg']

        static_loss = torch.tensor(0, device=static_pred.device)
        dynamic_loss = torch.tensor(0, device=dynamic_pred.device)

        # during training, we only need to compute the ego vehicle's gt loss
        static_gt = gt_dict['gt_static']
        dynamic_gt = gt_dict['gt_dynamic']
        static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
        dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

        if self.target == 'dynamic':
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)

        elif self.target == 'static':
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        else:
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss
        self.loss_dict.update({'total_loss': total_loss,
                               'static_loss': static_loss,
                               'dynamic_loss': dynamic_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                " || Dynamic Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), static_loss.item(), dynamic_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                  " || Dynamic Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), static_loss.item(), dynamic_loss.item()))


        writer.add_scalar('Static_loss', static_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                          epoch*batch_len + batch_id)




