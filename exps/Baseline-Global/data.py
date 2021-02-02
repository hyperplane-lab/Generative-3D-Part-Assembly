"""
    PartNetPartDataset
"""

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import ipdb


class PartNetPartDataset(data.Dataset):

    def __init__(self, category, data_dir, data_fn, data_features, level,\
            max_num_part=20):
        # store parameters
        self.data_dir = data_dir        # a data directory inside [path/to/codebase]/data/
        self.data_fn = data_fn          # a .npy data indexing file listing all data tuples to load
        self.category = category
        self.max_num_part = max_num_part
        self.max_pairs = max_num_part * (max_num_part-1) / 2
        self.level = level

        # load data
        self.data = np.load(os.path.join(self.data_dir, data_fn))

        # data features
        self.data_features = data_features

        self.part_sems = []
        self.part_sem2id = dict()


    def get_part_count(self):
        return len(self.part_sems)

    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_dir: %s, data_fn: %s, max_num_part: %d' % \
                (self.category, len(self), self.data_dir, self.data_fn, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        shape_id = self.data[index]
        #cur_data_fn = os.path.join(self.data_dir, '%s_level' % shape_id + self.level + '.npy')
        cur_data_fn = os.path.join(self.data_dir, 'shape_data/%s_level' % shape_id + self.level + '.npy')
        cur_contact_data_fn = os.path.join(self.data_dir, 'contact_points/pairs_with_contact_points_%s_level' % shape_id + self.level + '.npy')
        cur_data = np.load(cur_data_fn, allow_pickle=True ).item()   # assume data is stored in seperate .npz file
        cur_contacts = np.load(cur_contact_data_fn,allow_pickle=True)
        #cur_contact_list_fn = os.path.join(self.data_dir,"contact_point_64list/pairs_with_contact_points_%s_level" % shape_id + self.level + ".npy")
        #cur_contact_list = np.load(cur_contact_list_fn,allow_pickle = True).item()
        data_feats = ()

        for feat in self.data_features:

            if feat == 'contact_points':
                cur_num_part = cur_contacts.shape[0]
                out = np.zeros((self.max_num_part,self.max_num_part,4), dtype=np.float32)
                out[:cur_num_part,:cur_num_part,:] = cur_contacts
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'sym':
                cur_sym = cur_data['sym']
                cur_part_ids = cur_data['geo_part_ids']                 # p
                cur_num_part = cur_sym.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                out = np.zeros((self.max_num_part, cur_sym.shape[1]), dtype=np.float32)
                out[:cur_num_part] = cur_sym
                out = torch.from_numpy(out).float().unsqueeze(0)    # p x 3
                data_feats = data_feats + (out,)

            elif feat == 'part_pcs':
                cur_pts = cur_data['part_pcs']                      # p x N x 3 (p is unknown number of parts for this shape)
                cur_part_ids = cur_data['geo_part_ids']                 # p
                cur_num_part = cur_pts.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                out = np.zeros((self.max_num_part, cur_pts.shape[1], 3), dtype=np.float32)
                out[:cur_num_part] = cur_pts
                out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 x N x 3
                data_feats = data_feats + (out,)

            elif feat == 'part_poses':
                cur_pose = cur_data['part_poses']                   # p x (3 + 4)
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                out = np.zeros((self.max_num_part, 3 + 4), dtype=np.float32)
                out[:cur_num_part] = cur_pose
                out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 x (3 + 4)
                data_feats = data_feats + (out,)

            elif feat == 'part_valids':
                cur_pose = cur_data['part_poses']                   # p x (3 + 4)
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                out = np.zeros((self.max_num_part), dtype=np.float32)
                out[:cur_num_part] = 1
                out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 (return 1 for the first p parts, 0 for the rest)
                data_feats = data_feats + (out,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'part_ids':
                cur_part_ids = cur_data['geo_part_ids']
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids
                out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20
                data_feats = data_feats + (out,)

            elif feat == 'match_ids':
                cur_part_ids = cur_data['geo_part_ids']
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids
                index = 1
                for i in range(1,58):
                    idx = np.where(out==i)[0]
                    idx = torch.from_numpy(idx)
                    # print(idx)
                    if len(idx)==0: continue
                    elif len(idx)==1: out[idx]=0
                    else:
                        out[idx] = index
                        index += 1

                data_feats = data_feats + (out,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

