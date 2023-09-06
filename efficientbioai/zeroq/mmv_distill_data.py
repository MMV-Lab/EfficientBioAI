#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from utils import *


def own_loss(A, B):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
	"""
    return (A - B).norm()**2 / B.size(0)

class output_hook(object):
    """
	Forward_hook used to get the input of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def getDistilData(teacher_model,size):
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.

	teacher_model: pretrained single-precision model
	"""

    # initialize distilled data with random noise according to the dataset
    gaussian_data = (torch.randint(0,high=255, size=size).float() -
                  127.5) / 5418.75

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model.get_model().net.eval()
    
    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.get_model().net.modules()
    ])

    for n, m in teacher_model.get_model().net.named_modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d)) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten()))
    assert len(hooks) == len(bn_stats)

    
    # initialize the criterion, optimizer, and scheduler
    crit = nn.CrossEntropyLoss()
    gaussian_data.requires_grad = True
    params_dict = [{'params': gaussian_data, 'lr': 0.5}]
    optimizer = optim.Adam(params_dict)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)

    input_mean = torch.zeros(1, 1)
    input_std = torch.ones(1, 1)

    for it in range(1):
        teacher_model.get_model().net.zero_grad()
        optimizer.zero_grad()
        for hook in hooks:
            hook.clear()
        # gaussian_data_np = gaussian_data.cpu().detach().numpy()
        output = teacher_model.core_infer(gaussian_data)
        mean_loss = 0
        std_loss = 0

        # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
        for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
            tmp_output = hook.outputs
            tmp_output = hook.outputs
            bn_mean, bn_std = bn_stat[0], bn_stat[1]
            tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
            tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
            mean_loss += own_loss(bn_mean, tmp_mean)
            std_loss += own_loss(bn_std, tmp_std)
        tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 1,
                                                     -1),
                                  dim=2)
        tmp_std = torch.sqrt(
                torch.var(gaussian_data.view(gaussian_data.size(0), 1, -1),
                          dim=2) + eps)
        mean_loss += own_loss(input_mean, tmp_mean)
        std_loss += own_loss(input_std, tmp_std)
        total_loss = mean_loss + std_loss

        # update the distilled data
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

    refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian
