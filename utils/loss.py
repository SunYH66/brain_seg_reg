# --coding:utf-8--
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        """Computes the Dice loss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            smooth: added to the denominator for numerical stability.
        Returns:
            dice coefficient: the average 2 * class intersection over cardinality value
            for multi-class image segmentation
        """
        num_classes = int(targets.max()+1) # TODO: inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]

        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        # probas = F.softmax(inputs, dim=1)
        probas = inputs
        if probas.size(1) == 1:
            probas = torch.eye(num_classes)[inputs.squeeze(1).long()]
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice = torch.mean((2. * intersection + smooth) / (cardinality + smooth))

        return torch.tensor(1, dtype=torch.float, device='cuda') - dice


class ThreeDiceLoss(nn.Module):
    def __init__(self):
        super(ThreeDiceLoss, self).__init__()

    def forward(self, inputs_1, inputs_2, targets, smooth=1e-8):
        """Computes the Dice loss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs_1: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            inputs_2: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
            smooth: added to the denominator for numerical stability.
        Returns:
            dice coefficient: the average 2 * class intersection over cardinality value
            for multi-class image segmentation
        """
        num_classes = 4 # inputs_1.size(1)
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]

        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        true_1_hot = true_1_hot.type(inputs_1.type())
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(inputs_1 * inputs_2 * true_1_hot, dims)
        cardinality = torch.sum(inputs_1 + inputs_2 + true_1_hot, dims)
        dice = torch.mean((3. * intersection + smooth) / (cardinality + smooth))

        return torch.tensor(1, dtype=torch.float, device='cuda') - dice


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) / class_num
        else:
            assert len(alpha) == class_num
            self.alpha = torch.tensor(alpha)
            self.alpha = self.alpha.unsqueeze(1)
            self.alpha = self.alpha/self.alpha.sum()

        self.alpha = self.alpha.cuda()
        self.gamma = gamma
        self.size_average = size_average
        self.class_num = class_num
        self.one_hot_codes = torch.eye(self.class_num).cuda()
        # self.one_hot_codes = torch.eye(self.class_num)

    def forward(self, inputs, target):
        # the inputs size should be one of the follows
        # 1. B * class_num
        # 2. B, class_num, x, y
        # 3. B, class_num, x, y, z
        assert inputs.dim() == 2 or inputs.dim() == 4 or inputs.dim() == 5
        if inputs.dim() == 4:
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            inputs = inputs.view(inputs.numel()//self.class_num, self.class_num)
        elif inputs.dim() == 5:
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
            inputs = inputs.view(inputs.numel()//self.class_num, self.class_num)

        # the size of target tensor should be
        # 1. B, 1 or B
        # 2. B, 1, x, y or B, x, y
        # 3. B, 1, x, y, z or B, x, y, z
        target = target.contiguous()
        target = target.long().view(-1)

        mask = self.one_hot_codes[target.data]
        # mask = Variable(mask, requires_grad=False)

        alpha = self.alpha[target.data]
        # alpha = Variable(alpha, requires_grad=False)

        probs = (inputs*mask).sum(1).view(-1, 1) + 1e-10
        log_probs = probs.log()

        if self.gamma > 0:
            # batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_probs
            batch_loss = -alpha * (torch.tensor(1, dtype=torch.float, device='cuda') - probs) ** self.gamma * log_probs
        else:
            batch_loss = -alpha * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
