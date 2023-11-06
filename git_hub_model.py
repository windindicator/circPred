#!/usr/bin/python
from torch import nn
import torch


def prediction(model, sequence):
    sequence = model(sequence)
    label_index = torch.argmax(sequence, dim=-1)
    return label_index


# 搭建神经网络
class classification_model(nn.Module):
    def __init__(self):
        super(classification_model, self).__init__()
        self.conv_input = nn.Conv1d(1, 1100, kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_1 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_2 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_3 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_4 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_5 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_in_residual_1 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_in_residual_2 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_in_residual_3 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=3)
        self.conv_in_residual_4 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=9)
        self.conv_in_residual_5 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=27)
        self.batch_norm_1_1 = nn.BatchNorm1d(1100)
        self.batch_norm_1_2 = nn.BatchNorm1d(550)
        self.batch_norm_2_1 = nn.BatchNorm1d(1100)
        self.batch_norm_2_2 = nn.BatchNorm1d(550)
        self.batch_norm_3_1 = nn.BatchNorm1d(1100)
        self.batch_norm_3_2 = nn.BatchNorm1d(550)
        self.batch_norm_4_1 = nn.BatchNorm1d(1100)
        self.batch_norm_4_2 = nn.BatchNorm1d(550)
        self.batch_norm_5_1 = nn.BatchNorm1d(1100)
        self.batch_norm_out = nn.BatchNorm1d(1100)
        self.batch_norm_5_2 = nn.BatchNorm1d(550)
        self.sig_fn = nn.Sigmoid()
        self.relu_fn = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(640 * 1100, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        sequence = data
        sequence = self.conv_input(sequence)

        # residual block 1
        residual = self.batch_norm_1_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_1(residual)
        residual = self.batch_norm_1_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_1(residual)
        sequence = sequence + residual

        # residual block 2
        residual = self.batch_norm_2_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_2(residual)
        residual = self.batch_norm_2_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_2(residual)
        sequence = sequence + residual

        # residual block 3
        residual = self.batch_norm_3_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_3(residual)
        residual = self.batch_norm_3_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_3(residual)
        sequence = sequence + residual

        # residual block 4
        residual = self.batch_norm_4_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_4(residual)
        residual = self.batch_norm_4_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_4(residual)
        sequence = sequence + residual

        # residual block 5
        residual = self.batch_norm_5_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_5(residual)
        residual = self.batch_norm_5_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_5(residual)
        sequence = sequence + residual

        sequence = self.flatten(sequence)
        sequence = self.linear1(sequence)
        return sequence


class scanning_model(nn.Module):
    def __init__(self):
        super(scanning_model, self).__init__()
        self.conv_input = nn.Conv1d(640, 1100, kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_1 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_2 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_3 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_4 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_to_get_residue_5 = nn.Conv1d(550, 1100,
                                               kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_in_residual_1 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_in_residual_2 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_in_residual_3 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=3)
        self.conv_in_residual_4 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=9)
        self.conv_in_residual_5 = nn.Conv1d(1100, 550,
                                            kernel_size=3, stride=1, padding='same', dilation=27)
        self.last_conv = nn.Conv1d(1100, 1,
                                   kernel_size=1, stride=1, padding='same')
        self.mean_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding='same')
        self.batch_norm_1_1 = nn.BatchNorm1d(1100)
        self.batch_norm_1_2 = nn.BatchNorm1d(550)
        self.batch_norm_2_1 = nn.BatchNorm1d(1100)
        self.batch_norm_2_2 = nn.BatchNorm1d(550)
        self.batch_norm_3_1 = nn.BatchNorm1d(1100)
        self.batch_norm_3_2 = nn.BatchNorm1d(550)
        self.batch_norm_4_1 = nn.BatchNorm1d(1100)
        self.batch_norm_4_2 = nn.BatchNorm1d(550)
        self.batch_norm_5_1 = nn.BatchNorm1d(1100)
        self.batch_norm_5_2 = nn.BatchNorm1d(550)
        self.sig_fn = nn.Sigmoid()
        self.relu_fn = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1000, 1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        sequence = data
        sequence = self.conv_input(sequence)

        # residual block 1
        residual = self.batch_norm_1_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_1(residual)
        residual = self.batch_norm_1_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_1(residual)
        sequence = sequence + residual

        # residual block 2
        residual = self.batch_norm_2_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_2(residual)
        residual = self.batch_norm_2_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_2(residual)
        sequence = sequence + residual

        # residual block 3
        residual = self.batch_norm_3_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_3(residual)
        residual = self.batch_norm_3_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_3(residual)
        sequence = sequence + residual

        # residual block 4
        residual = self.batch_norm_4_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_4(residual)
        residual = self.batch_norm_4_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_4(residual)
        sequence = sequence + residual

        # residual block 5
        residual = self.batch_norm_5_1(sequence)
        residual = self.relu_fn(residual)
        residual = self.conv_in_residual_5(residual)
        residual = self.batch_norm_5_2(residual)
        residual = self.relu_fn(residual)
        residual = self.conv_to_get_residue_5(residual)
        sequence = sequence + residual

        sequence = self.last_conv(sequence)
        sequence = self.mean_conv(sequence)
        sequence = self.flatten(sequence)
        '''sequence = self.linear(sequence)'''
        return sequence
