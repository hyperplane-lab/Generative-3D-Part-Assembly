"""
    Scene Graph to predict the pose of each part
    adjust relation using the t in last iteration
    model v30: based on v18
    model v18: the same as v12
    model v12:

    model_v3: iter backwards.
    and 
    model v6: not share weights.
    and 
    model v9: replace mlp4 as GRU
    
    with local info

    Input:
        relation matrxi of parts,part valids, part point clouds, instance label, iter_ind, pred_part_poses:      B x P x P, B x P, B x P x N x 3, B x P x P , (1 or 2 or 3) , B x P x 7
    Output:
        R and T:                B x P x (3 + 4)
    Losses:
        Center L2 Loss, Rotation L2 Loss, Rotation Chamder-Distance Loss
    Setting 1:
        when passing MLP5, global feature is not explicitly concatted
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys, os
import ipdb
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from quaternion import qrot
import ipdb
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.join(BASE_DIR, '../../exp_GAN/models/sampling'))
#from sampling import furthest_point_sample



class MLP2(nn.Module):
    def __init__(self, feat_len):
        super(MLP2, self).__init__()

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
        Input: B x N x 3 (B x P x N x 3)
        Output: B x F (B x P x F)
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


class MLP3(nn.Module):
    def __init__(self, feat_len):
        super(MLP3, self).__init__()

        self.conv1 = nn.Conv1d(2*feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        #self.mlp1 = nn.Linear(2*feat_len, 512)
        #self.mlp2 = nn.Linear(512, 512)
        #self.mlp3 = nn.Linear(512, feat_len)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)

        #self.mlp1 = nn.Linear(512 + 512, feat_len)

    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):
        #num_part = x.shape[1]

        x = x.permute(0, 2, 1)
        # x = self.conv1(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class MLP4(nn.Module):
    def __init__(self, feat_len):
        super(MLP4, self).__init__()

        self.conv1 = nn.Conv1d(2*feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        #self.mlp1 = nn.Linear(2*feat_len, 512)
        #self.mlp2 = nn.Linear(512, 512)
        #self.mlp3 = nn.Linear(512, feat_len)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)

        #self.mlp1 = nn.Linear(512 + 512, feat_len)

    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):
        #num_part = x.shape[1]

        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class GRU(nn.Module):
    def __init__(self, feat_len, num_layer):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(2 * feat_len, feat_len, num_layer)

    """
        Input: B x P x (F + F), num_layer x B x F
        Output: B x P x F
    """

    def forward(self, x, h0):
        num_part = x.shape[1]
        batch_size = x.shape[0]

        x = x.view(1, batch_size * num_part, -1)
        output, hn = self.rnn(x, h0)
        output = output.view(batch_size, num_part, -1)

        return output, hn



class MLP5(nn.Module):

    def __init__(self, feat_len):
        super(MLP5, self).__init__()

        self.mlp = nn.Linear(feat_len, 512)

        self.trans = nn.Linear(512, 3)

        self.quat = nn.Linear(512, 4)
        self.quat.bias.data.zero_()

    """
        Input: * x F    (* denotes any number of dimensions, used as B x P here)
        Output: * x 7   (* denotes any number of dimensions, used as B x P here)
    """

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))

        trans = torch.tanh(self.trans(feat))  # consider to remove torch.tanh if not using PartNet normalization

        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat], dim=-1)
        return out

class R_Predictor(nn.Module):
    def __init__(self):
        super(R_Predictor, self).__init__()
        self.mlp1 = nn.Linear(128 + 128, 256)
        self.mlp2 = nn.Linear(256,512)
        self.mlp3 = nn.Linear(512,1)
        
    def forward(self, x):
        x = torch.relu(self.mlp1(x)) 
        x = torch.relu(self.mlp2(x)) 
        x = torch.sigmoid(self.mlp3(x)) 
        return x

class Pose_extractor(nn.Module):
    def __init__(self):
        super(Pose_extractor, self).__init__()
        self.mlp1 = nn.Linear(7, 256)
        self.mlp2 = nn.Linear(256,128)
        
    def forward(self, x):
        x = torch.relu(self.mlp1(x)) 
        x = torch.relu(self.mlp2(x)) 
        return x

class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf
        self.mlp2 = MLP2(conf.feat_len)
        self.mlp3_1 = MLP3(conf.feat_len)
        self.mlp3_2 = MLP3(conf.feat_len)
        self.mlp3_3 = MLP3(conf.feat_len)
        self.mlp3_4 = MLP3(conf.feat_len)
        self.mlp3_5 = MLP3(conf.feat_len)

        self.mlp4_1 = MLP4(conf.feat_len) 
        self.mlp4_2 = MLP4(conf.feat_len) 
        self.mlp4_3 = MLP4(conf.feat_len) 
        self.mlp4_4 = MLP4(conf.feat_len) 
        self.mlp4_5 = MLP4(conf.feat_len) 

        self.mlp5_1 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)
        self.mlp5_2 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)
        self.mlp5_3 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)
        self.mlp5_4 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)
        self.mlp5_5 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)

        self.relation_predictor = R_Predictor()
        self.relation_predictor_dense = R_Predictor()
        self.pose_extractor = Pose_extractor()
    """
        Input: B x P x P, B x P, B x P x N x 3, B x P x P
        Output: B x P x (3 + 4)
    """

    def forward(self, conf, relation_matrix, part_valids, part_pcs, instance_label, class_list):
        batch_size = part_pcs.shape[0]
        num_part = part_pcs.shape[1]
        relation_matrix = relation_matrix.double()
        valid_matrix = copy.copy(relation_matrix)
        pred_poses = torch.zeros((batch_size, num_part, 7)).to(conf.device)
        total_pred_poses = []
        # obtain per-part feature
        part_feats = self.mlp2(part_pcs.view(batch_size * num_part, -1, 3)).view(batch_size, num_part, -1)  # output: B x P x F
        local_feats = part_feats
        random_noise = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, num_part, 16]).astype(
            np.float32)  # B x P x 16
        random_noise = torch.tensor(random_noise).to(self.conf.device)  # B x P x 16
        
        for iter_ind in range(self.conf.iter):
            # adjust relations
            if iter_ind >= 1 :
                cur_poses = copy.copy(pred_poses).double()            
                pose_feat = self.pose_extractor(cur_poses.float())
                if iter_ind % 2 == 1: 
                    for i in range(batch_size):
                        for j in range(len(class_list[i])):
                            cur_pose_feats = pose_feat[i,class_list[i][j]]
                            cur_pose_feat = cur_pose_feats.max(dim = -2)[0] 
                            pose_feat[i,class_list[i][j]]=cur_pose_feat
                            part_feats_copy = copy.copy(part_feats)
                            with torch.no_grad():
                                part_feats_copy[i,class_list[i][j]] = part_feats_copy[i, class_list[i][j]].max(dim = -2)[0]

                pose_featA = pose_feat.unsqueeze(1).repeat(1,num_part,1,1)
                pose_featB = pose_feat.unsqueeze(2).repeat(1,1,num_part,1)
                input_relation = torch.cat([pose_featA,pose_featB],dim = -1).float()
                if iter_ind % 2 == 0:
                    new_relation = self.relation_predictor_dense(input_relation.view(batch_size,-1,256)).view(batch_size,num_part,num_part)
                elif iter_ind % 2 == 1:
                    new_relation = self.relation_predictor(input_relation.view(batch_size,-1,256)).view(batch_size,num_part,num_part)
                relation_matrix = new_relation.double() * valid_matrix 
            # mlp3
            if iter_ind>=1 and iter_ind%2==1: 
                part_feat1 = part_feats_copy.unsqueeze(2).repeat(1, 1, num_part, 1) # B x P x P x F
                part_feat2 = part_feats_copy.unsqueeze(1).repeat(1, num_part, 1, 1) # B x P x P x F
            else:
                part_feat1 = part_feats.unsqueeze(2).repeat(1, 1, num_part, 1) # B x P x P x F
                part_feat2 = part_feats.unsqueeze(1).repeat(1, num_part, 1, 1) # B x P x P x F
            input_3 = torch.cat([part_feat1, part_feat2], dim=-1) # B x P x P x 2F
            if iter_ind == 0:
                mlp3 = self.mlp3_1
                mlp4 = self.mlp4_1
                mlp5 = self.mlp5_1
            elif iter_ind == 1:
                mlp3 = self.mlp3_2
                mlp4 = self.mlp4_2
                mlp5 = self.mlp5_2
            elif iter_ind == 2:
                mlp3 = self.mlp3_3
                mlp4 = self.mlp4_3
                mlp5 = self.mlp5_3
            elif iter_ind == 3:
                mlp3 = self.mlp3_4
                mlp4 = self.mlp4_4
                mlp5 = self.mlp5_4
            elif iter_ind == 4:
                mlp3 = self.mlp3_5
                mlp4 = self.mlp4_5
                mlp5 = self.mlp5_5
            # for the pair of parts (A, B), A is the query one, A is about the row, A is the former in part_feats
            part_relation = mlp3(input_3.view(batch_size * num_part, num_part, -1)).view(batch_size, num_part,
                                     num_part, -1) # B x P x P x F

            # pooling
            part_message = part_relation.double() * relation_matrix.unsqueeze(3).double() # B x P x P x F
            part_message = part_message.sum(dim=2) # B x P x F
            norm = relation_matrix.sum(dim=-1) # B x P
            delta = 1e-6
            normed_part_message = part_message / (norm.unsqueeze(dim=2) + delta) # B x P x F

            # mlp4
            input_4 = torch.cat([normed_part_message.double(), part_feats.double()], dim=-1) # B x P x 2F
            part_feats= mlp4(input_4.float()) # B x P x F
            
            # mlp5
            input_5 = torch.cat([local_feats, part_feats.float(), instance_label, pred_poses, random_noise],dim=-1)
            pred_poses = mlp5(input_5.float())

            # save poses 
            total_pred_poses.append(pred_poses)

        return total_pred_poses


    """
            Input: * x N x 3, * x 3, * x 4, * x 3, * x 4,
            Output: *, *  (two lists)
    """

    def linear_assignment(self, pts, centers1, quats1, centers2, quats2):
        import random
        pts_to_select = torch.tensor(random.sample([i for i  in range(1000)],100))
        pts = pts[:,pts_to_select] 
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
        return loss_per_data  
        
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

        """
            output : B
        """
    def get_sym_point(self, point, x, y, z):

        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

        return point.tolist()

    def get_possible_point_list(self, point, sym):
        sym = torch.tensor([1.0,1.0,1.0]) 
        point_list = []
        #sym = torch.tensor(sym)
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
        #print(list1[0])
        #print(list2[0])
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
        #ipdb.set_trace()
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
            #print("Shape id is", b)
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


        #print(count, total_num)
        return contact_point_loss, count, total_num

