# B-LSTM to directly regress 3 translation + 4 quatanian

import torch
from torch import nn
import torch.nn.functional as F
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from quaternion import qrot
import ipdb
import random
import numpy as np
import operator

from scipy.optimize import linear_sum_assignment


class PartPointNet(nn.Module):
    def __init__(self, feat_len):
        super(PartPointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(1024, feat_len)
        self.bn6 = nn.BatchNorm1d(feat_len)

    """
        Input: B x N x 3
        Output: B x F
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x


class PoseDecoder(nn.Module):

    def __init__(self, feat_len):
        super(PoseDecoder, self).__init__()

        self.mlp = nn.Linear(feat_len, 512)

        self.trans = nn.Linear(512, 3)

        self.quat = nn.Linear(512, 4)
        self.quat.bias.data.zero_()

    """
        Input: * x F    (* denotes any number of dimensions)
        Output: * x 7   (* denotes any number of dimensions)
    """

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))

        trans = torch.tanh(self.trans(feat))  # consider to remove torch.tanh if not using PartNet normalization

        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat], dim=-1)
        return out

# RNN, inspired by CVPR 2020 PQ-Net, partly from PQ-Net
##############################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)

        self.init_hidden = self.initHidden()

    def forward(self, input, init_hidden):
        """
        :param input: (seq_len, batch_size, feature_dim)
        :return:
            output: (seq_len, batch, num_directions * hidden_size)
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        output, hidden = self.gru(input, init_hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(self.n_layer * self.num_directions, batch_size, self.hidden_size, requires_grad=False)


#inspired by CVPR 2020, PQ-Net, partly from PQ-Net
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.n_units_hidden1 = 256
        self.n_units_hidden2 = 128

        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)
        self.linear1 = nn.Sequential(nn.Linear(hidden_size, self.n_units_hidden1),
                                     nn.LeakyReLU(True),
                                     nn.Linear(self.n_units_hidden1, input_size),
                                     # nn.Sigmoid()
                                     )
        self.linear2 = nn.Sequential(nn.Linear(hidden_size, self.n_units_hidden2),
                                     nn.ReLU(True),
                                     nn.Dropout(0.2),
                                     nn.Linear(self.n_units_hidden2, 6),
                                     # nn.Sigmoid()
                                     )
        self.linear3 = nn.Sequential(nn.Linear(hidden_size, self.n_units_hidden2),
                                     nn.ReLU(True),
                                     nn.Dropout(0.2),
                                     nn.Linear(self.n_units_hidden2, 1),
                                     # nn.Sigmoid()
                                     )

        self.lockdrop = LockedDropout()
        self.dropout_i = 0.2
        self.dropout_o = 0.2

        self.init_input = self.initInput()

    def forward(self, input, hidden):
        """
        :param input: (1, batch, output_size)
        :param hidden: initial hidden state
        :return:
            output: (1, batch, num_directions * hidden_size)
            hidden: (num_layers * 1, batch, hidden_size)
            output_seq: (batch, 1 * output_size)
            stop_sign: (batch, 1)
        """
        # seq_len, batch_size = input.size(0), input.size(1)
        input = self.lockdrop(input, self.dropout_i)
        output, hidden = self.gru(input, hidden)
        # hidden : (num_layers * 1, batch, hidden_size)
        hidden1, hidden2 = torch.split(hidden, 1, 0)
        # output_ = self.lockdrop(output, self.dropout_o)
        output_code = self.linear1(hidden1.squeeze(0))
        # output_param = self.linear2(hidden2.squeeze(0))
        stop_sign = self.linear3(hidden1.squeeze(0))
        # output_seq = torch.cat([output_code, output_param], dim=1)
        output_seq = output_code

        return output, hidden, output_seq, stop_sign

    def initInput(self):
        # initial_code = torch.zeros((1, 1, self.input_size - 6), requires_grad=False)
        # initial_param = torch.tensor([0.5, 0.5, 0.5, 1, 1, 1], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)
        # initial = torch.cat([initial_code, initial_param], dim=2)
        initial = torch.zeros((1, 1, self.input_size), requires_grad=False)
        return initial


# Seq2Seq, inspired by CVPR 2020 PQ-Net, partly from PQ-Net
##############################################################################
class Seq2SeqAE(nn.Module):
    def __init__(self, en_input_size, de_input_size, hidden_size, conf):
        super(Seq2SeqAE, self).__init__()
        self.n_layer = 2
        self.encoder = EncoderRNN(en_input_size, hidden_size, n_layer=self.n_layer, bidirectional=True)
        self.decoder = DecoderRNN(de_input_size, hidden_size * 2 + 16, n_layer=self.n_layer, bidirectional=False)
        self.max_length = 10
        self.conf = conf

    def infer_encoder(self, input_seq, batch_size=1):
        """
        :param input_seq: (n_parts, 1, feature_dim)
        :return:
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        encoder_init_hidden = self.encoder.init_hidden.repeat(1, batch_size, 1).cuda()
        _, hidden = self.encoder(input_seq, encoder_init_hidden)
        hidden = hidden.view(self.n_layer, 2, batch_size, -1)
        hidden0, hidden1 = torch.split(hidden, 1, 1)
        hidden = torch.cat([hidden0.squeeze(1), hidden1.squeeze(1)], 2)
        # hidden = hidden[-1]
        return hidden

    def infer_decoder(self, decoder_hidden, target_seq, teacher_forcing_ratio=0.5):
        batch_size = target_seq.size(1)
        target_length = target_seq.size(0)
        decoder_input = self.decoder.init_input.detach().repeat(1, batch_size, 1).cuda()

        # Teacher forcing: Feed the target as the next input
        # Without teacher forcing: use its own predictions as the next input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        stop_signs = []
        for di in range(target_length):
            decoder_output, decoder_hidden, output_seq, stop_sign = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(output_seq)
            stop_signs.append(stop_sign)
            # using target seq as input or not
            decoder_input = target_seq[di:di+1] if use_teacher_forcing else output_seq.detach().unsqueeze(0)
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        stop_signs = torch.stack(stop_signs, dim=0)
        return decoder_outputs, stop_signs

    def infer_decoder_stop(self, decoder_hidden, length=None):
        decoder_outputs = []
        stop_signs = []
        decoder_input = self.decoder.init_input.detach().repeat(1, 1, 1).cuda()
        for di in range(self.max_length):
            decoder_output, decoder_hidden, output_seq, stop_sign = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(output_seq)
            stop_signs.append(stop_sign)
            if length is not None:
                if di == length - 1:
                    break
            elif torch.sigmoid(stop_sign[0, 0]) > 0.5:
                # stop condition
                break
            decoder_input = output_seq.detach().unsqueeze(0)  # using target seq as input
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        stop_signs = torch.stack(stop_signs, dim=0)
        return decoder_outputs, stop_signs

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        """
        :param input_seq: (seq_len, batch_size, feature_dim) PackedSequence
        :param target_seq: (seq_len, batch_size, feature_dim)
        :param teacher_forcing_ratio: float
        :return:
            decoder_outputs: (seq_len, batch, num_directions, output_size)
            stop_signs: (seq_len, batch, num_directions, 1)
        """
        batch_size = target_seq.size(1)
        # create random noise
        random_noise = np.random.normal(loc=0.0, scale=1.0, size=[self.n_layer * 1, batch_size, 16]).astype(
            np.float32)  # n_layer x B x 16
        random_noise = torch.tensor(random_noise).to(self.conf.device)

        encoder_hidden = self.infer_encoder(input_seq, batch_size)
        # ipdb.set_trace()
        decoder_hidden = torch.cat([encoder_hidden, random_noise], dim=2) # .unsqueeze(0)
        decoder_outputs, stop_signs = self.infer_decoder(decoder_hidden, target_seq, teacher_forcing_ratio)
        return decoder_outputs, stop_signs


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf

        self.seq2seqae = Seq2SeqAE(conf.feat_len, conf.feat_len, conf.hidden_size, conf)

        self.encoder = PartPointNet(conf.feat_len)
        self.decoder = PoseDecoder(conf.feat_len)

    """
        Input: B x P x N x 3, B x P
        Output: B x P x (3 + 4)
    """

    def forward(self, part_pcs, part_valids):
        batch_size = part_pcs.shape[0]
        num_part = part_pcs.shape[1]

        # obtain per-part feature
        part_feats = self.encoder(part_pcs.view(batch_size * num_part, -1, 3)).view(batch_size, num_part,
                                                                                          -1)  # B x P x F
        part_feature_seq = part_feats.transpose(0, 1)
        target_seq = part_feature_seq.detach()
        output_seq, output_stop = self.seq2seqae(part_feature_seq, target_seq)   # P x B x num_directions(1) x F
        output_seq = output_seq.squeeze(2).transpose(0, 1)  # B x P x F

        # compute pred poses
        pred_poses = self.decoder(output_seq)
        return pred_poses

    """
        Input: B x P x 3, B x P x 3, B x P
        Output: B
    """

    def get_trans_l2_loss(self, trans1, trans2, valids):
        loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """

    def get_rot_l2_loss(self, pts, quat1, quat2, valids):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """

    def get_rot_cd_loss(self, pts, quat1, quat2, valids, device):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)

        loss_per_data = loss_per_data.to(device)
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data  #

    def get_total_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
        center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)

        thre = 0.01
        loss_per_data = loss_per_data.to(device)
        acc = [[0 for i in range(num_part)] for j in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_part):
                if loss_per_data[i, j] < thre and valids[i, j]:
                    acc[i][j] = 1
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data, acc

    def get_shape_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
        center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        pts1 = pts1.view(batch_size, num_part * num_point, 3)
        pts2 = pts2.view(batch_size, num_part * num_point, 3)
        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        valids = valids.unsqueeze(2).repeat(1, 1, 1000).view(batch_size, -1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

        loss_per_data = loss_per_data.to(device)
        return loss_per_data

    """
            output : B
    """

    def get_resample_shape_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        part_sum = valids.sum(-1)  # B
        center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
        center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        shape_pcs1 = pts1.view(batch_size, -1, 3)  # B x PN x 3
        sample_pcs1 = torch.zeros(batch_size, 2048, 3).to(device)

        for index in range(batch_size):
            with torch.no_grad():
                shape_pc_id1 = torch.tensor([index]).unsqueeze(1).repeat(1, 2048).long().view(-1).to(device)
                shape_pc_id2 = furthest_point_sample(
                    shape_pcs1[index][:int(part_sum[index].item()) * num_point].unsqueeze(0), 2048).long().view(-1)
            sample_pcs1[index] = shape_pcs1[shape_pc_id1, shape_pc_id2]

        shape_pcs2 = pts2.view(batch_size, -1, 3)  # B x PN x 3
        sample_pcs2 = torch.zeros(batch_size, 2048, 3).to(device)

        for index in range(batch_size):
            with torch.no_grad():
                shape_pc_id1 = torch.tensor([index]).unsqueeze(1).repeat(1, 2048).long().view(-1).to(device)
                shape_pc_id2 = furthest_point_sample(
                    shape_pcs2[index][:int(part_sum[index].item()) * num_point].unsqueeze(0), 2048).long().view(-1)
            sample_pcs2[index] = shape_pcs2[shape_pc_id1, shape_pc_id2]

        dist1, dist2 = chamfer_distance(sample_pcs1, sample_pcs2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

        loss_per_data = loss_per_data.to(device)
        return loss_per_data


    def get_sym_point(self, point, x, y, z):

        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

        return point.tolist()

    def get_possible_point_list(self, point, sym):
        point_list = []
        if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
        elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
        elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
        elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
        elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
        else:
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 1))

        return point_list

    def get_min_l2_dist(self, list1, list2, center1, center2, quat1, quat2):

        list1 = torch.tensor(list1) # m x 3
        list2 = torch.tensor(list2) # n x 3
        len1 = list1.shape[0]
        len2 = list2.shape[0]
        center1 = center1.unsqueeze(0).repeat(len1, 1)
        center2 = center2.unsqueeze(0).repeat(len2, 1)
        quat1 = quat1.unsqueeze(0).repeat(len1, 1)
        quat2 = quat2.unsqueeze(0).repeat(len2, 1)
        list1 = list1.to(self.conf.device)
        list2 = list2.to(self.conf.device)
        list1 = center1 + qrot(quat1, list1)
        list2 = center2 + qrot(quat2, list2)
        mat1 = list1.unsqueeze(1).repeat(1, len2, 1)
        mat2 = list2.unsqueeze(0).repeat(len1, 1, 1)
        mat = (mat1 - mat2) * (mat1 - mat2)
        mat = mat.sum(dim=-1)
        return mat.min()

    """    
        Contact point loss metric
        Date: 2020/5/22
        Input B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput B
    """
    def get_contact_point_loss(self, center, quat, contact_points, sym_info):

        batch_size = center.shape[0]
        num_part = center.shape[1]
        contact_point_loss = torch.zeros(batch_size)
        total_num = 0
        count = 0
        for b in range(batch_size):
            sum_loss = 0
            for i in range(num_part):
                for j in range(num_part):
                    if contact_points[b, i, j, 0]:
                        contact_point_1 = contact_points[b, i, j, 1:]
                        contact_point_2 = contact_points[b, j, i, 1:]
                        sym1 = sym_info[b, i]
                        sym2 = sym_info[b, j]
                        point_list_1 = self.get_possible_point_list(contact_point_1, sym1)
                        point_list_2 = self.get_possible_point_list(contact_point_2, sym2)
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                        #print(dist)
                        if dist < 0.01:
                            count += 1
                        total_num += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss

        return contact_point_loss, count, total_num


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.detach().clone().requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


if __name__ == '__main__':
    pass
