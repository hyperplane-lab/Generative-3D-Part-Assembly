"""
    B-Complement
    Input:
        part point clouds:      B x P x N x 3
    Output:
        R and T:                B x P x(3 + 4)
    Losses:
        Center L2 Loss, Rotation L2 Loss, Rotation Chamder-Distance Loss
"""

import torch
from torch import nn
import torch.nn.functional as F
import sys, os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from quaternion import qrot
import ipdb
from scipy.optimize import linear_sum_assignment


# PointNet Front-end
class PartPointNet(nn.Module):
    def __init__(self, feat_len):
        super(PartPointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        #self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        #self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(128, feat_len)
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
        #x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x


# PointNet Back-end
class PoseDecoder(nn.Module):

    def __init__(self, feat_len):
        super(PoseDecoder, self).__init__()

        self.mlp1 = nn.Linear(feat_len, 512)
        self.mlp2 = nn.Linear(512, 256)

        self.trans = nn.Linear(256, 3)

        self.quat = nn.Linear(256, 4)
        self.quat.bias.data.zero_()

    """
        Input: B x (2F + P + 16)    
        Output: B x 7   
    """

    def forward(self, feat):
        feat = torch.relu(self.mlp1(feat))
        feat = torch.relu(self.mlp2(feat))

        trans = torch.tanh(self.trans(feat))  # consider to remove torch.tanh if not using PartNet normalization

        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat.squeeze(0)], dim=-1)
        return out


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf

        self.part_pointnet = PartPointNet(conf.feat_len)

        self.pose_decoder = PoseDecoder(2 * conf.feat_len + conf.max_num_part + 16)

    """
        Input: B x P x N x 3, B x P, B x P x P, B x 7
        Output: B x P x (3 + 4)
    """

    def forward(self,seq, part_pcs, part_valids, instance_label, gt_part_pose):
        batch_size = part_pcs.shape[0]
        num_part = part_pcs.shape[1]
        num_point = part_pcs.shape[2]
        pred_part_poses = np.zeros((batch_size, num_part, 7))
        pred_part_poses = torch.tensor(pred_part_poses).to(self.conf.device)
        # generate random_noise
        random_noise = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, num_part, 16]).astype(
            np.float32)  # B x P x 16
        random_noise = torch.tensor(random_noise).to(self.conf.device)

        for iter in range(num_part):
            select_ind  = seq[:,iter].int().tolist()
            batch_ind = [i for i in range(len(select_ind))]

            if iter == 0:
                cur_pred_pose = gt_part_pose   # B x 7
                pred_part_poses= pred_part_poses.float()
                pred_part_poses[batch_ind,select_ind,:] = cur_pred_pose
                cur_pred_center = cur_pred_pose[:, :3].unsqueeze(1).repeat(1, num_point, 1) # B x N x 3
                cur_pred_qrot = cur_pred_pose[:, 3:].unsqueeze(1).repeat(1, num_point, 1)  # B x N x 4
                cur_part = cur_pred_center + qrot(cur_pred_qrot, part_pcs[batch_ind,select_ind, :, :])# B x N x 3
                cur_part = cur_part.unsqueeze(1)   # B x 1 x N x 3
                cur_shape = cur_part    # B x batch_ind,select_ind x N x 3
            else:
                cur_shape_feat = self.part_pointnet(cur_shape.view(batch_size, -1, 3))  # B x F
                cur_part_feat = self.part_pointnet(part_pcs[batch_ind,select_ind, :, :])# B x F
                cat_feat = torch.cat([cur_shape_feat, cur_part_feat, instance_label[batch_ind,select_ind, :].contiguous(), random_noise[batch_ind,select_ind, :].contiguous()], dim=-1)  # B x (2F + P + 16)
                cur_pred_pose = self.pose_decoder(cat_feat)   # B x 7
                pred_part_poses[batch_ind,select_ind, :] = cur_pred_pose
                cur_pred_center = cur_pred_pose[:, :3].unsqueeze(1).repeat(1, num_point, 1)  # B x N x 3
                cur_pred_qrot = cur_pred_pose[:, 3:].unsqueeze(1).repeat(1, num_point, 1)  # B x N x 4
                cur_part = cur_pred_center + qrot(cur_pred_qrot, part_pcs[batch_ind,select_ind, :, :])  # B x N x 3
                cur_part = cur_part.unsqueeze(1)  # B x 1 x N x 3
                cur_shape = torch.cat([cur_shape, cur_part], dim=1)  # B x select_ind x N x 3

        pred_part_poses = pred_part_poses.double() * part_valids.unsqueeze(2).double()

        return pred_part_poses.float()


    """  
            Input: * x N x 3, * x 3, * x 4, * x 3, * x 4,
            Output: *, *  (two lists)
    """

    def linear_assignment(self, pts, centers1, quats1, centers2, quats2):

        cur_part_cnt = pts.shape[0]
        num_point = pts.shape[1]

        with torch.no_grad():

            cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

            cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

            cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
            cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
            dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
            dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
            rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

        return rind, cind


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
        num_part =  pts.shape[1]
        num_point = pts.shape[2]

        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)
        
        thre = 0.01
        loss_per_data = loss_per_data.to(device)
        acc = [[0 for i in range(num_part)]for j in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_part):
                if loss_per_data[i,j] < thre and valids[i,j]:
                    acc[i][j] = 1
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data , acc

    def get_shape_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        pts1 = pts1.view(batch_size,num_part*num_point,3)
        pts2 = pts2.view(batch_size,num_part*num_point,3)
        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        valids = valids.unsqueeze(2).repeat(1,1,1000).view(batch_size,-1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
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
        sym = torch.tensor([1.0, 1.0, 1.0])
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

        list1 = torch.tensor(list1)  # m x 3
        list2 = torch.tensor(list2)  # n x 3
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
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :],
                                                    quat[b, i, :], quat[b, j, :])  # 1
                        if dist < 0.01:
                            count += 1
                        total_num += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss

        return contact_point_loss, count, total_num
