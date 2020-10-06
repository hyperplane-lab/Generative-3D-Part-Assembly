'''
    prepare shape data 
    include:
        point clouds: P x N x 3
        part ids: P         # the class index in partnet dataset
        geo part ids: P     # the class index according to the geo information
        part poses: P x 7   # T and R
            T: center of a part
            R: quaternion
'''
import numpy as np
import json
import sys
import os
import json
import torch
import numpy as np
from torch.utils import data
from pyquaternion import Quaternion
from sklearn.decomposition import PCA
from collections import namedtuple
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, random_split
import trimesh
import json
import ipdb

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
'''
    input: 
        size P x 3
        gt_pcs P x 1000 x 3
    output:
        sym_stick P
'''

def get_pc_center(pc):
    return np.mean(pc, axis=0)

def get_pc_scale(pc):
    return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

def get_pca_axes(pc):
    axes = PCA(n_components=3).fit(pc).components_
    return axes

def get_chamfer_distance(pc1, pc2):
    dist = cdist(pc1, pc2)
    error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    scale = get_pc_scale(pc1) + get_pc_scale(pc2)
    return error / scale


def get_sym(gt_pcs): 
    thre_for_sym = 0.025
    num_part = len(gt_pcs)
    sym = np.zeros((num_part,3))
    for j in range(3):
            for i in range(num_part):
                sym_pcs = gt_pcs[i].copy()
                sym_pcs[:,j] *= -1
                error = get_chamfer_distance(gt_pcs[i], sym_pcs)
                if error < thre_for_sym :
                    sym[i,j] = 1 
    return sym
        
def get_sym_stick(sizes, gt_pcs): 
    thre_for_stick = 5
    thre_for_sym = 0.025
    is_stick = [0 for i in range(len(sizes))]
    is_sym_stick = [0 for i in range(len(sizes))]
    for i in range(len(sizes)):
        r = sizes[i][1]**2 + sizes[i][0]**2
        r = r**(0.5)
        if sizes[i][2]/r > thre_for_stick:
            is_stick[i] = 1
    for i in range(len(sizes)):
        if is_stick[i]:
            sym_pcs = gt_pcs[i].copy()
            # sym_pcs[:,2] = - gt_pcs[i,:,2] 
            sym_pcs[:,2] *= -1
            error = get_chamfer_distance(gt_pcs[i], sym_pcs)
            print(error)
            if error < thre_for_sym :
                is_sym_stick[i] = 1 
    return is_sym_stick
        


def get_shape_info(index, lev):
    # get children
    fn = "../data/partnet_dataset/" + str(index) + "/result_after_merging.json"
    root_to_load_file = []
    with open( fn, "r" ) as f:
        root_to_load_file = json.load(f)[0]
    parts_objs, parts_names = get_parts_objs(root_to_load_file, lev, '')
    
    parts_v = []
    parts_f = []
    for part_objs in parts_objs:
        obj_fns = ["../data/partnet_dataset/" + str(index) + "/objs/" + obj +'.obj' for obj in part_objs]
        vs=[]
        fs=[]
        for obj_fn in obj_fns:
            v,f = load_obj(obj_fn)
            vs.append(v)
            fs.append(f)
        v,f = merge_objs(vs,fs)
        parts_v.append(v)
        parts_f.append(f)
    parts_points, Rs, ts, sizes = get_norm_parts_points(parts_v, parts_f)

    return parts_points, Rs, ts, parts_names, sizes

def get_parts_objs(root_to_load_file, lev, root_to_load_file_name):
    try:
        children = root_to_load_file['children']
    except KeyError:
        return [root_to_load_file['objs']], [root_to_load_file_name + '/' + root_to_load_file['name']]
    parts_objs = []
    parts_names = []
    for child in children:
        if root_to_load_file_name + '/' + root_to_load_file['name'] + '/' + child['name'] in lev:
            parts_objs.append(child['objs'])
            parts_names.append(root_to_load_file_name + '/' + root_to_load_file['name'] + '/' + child['name'])
        else:
            child_objs, child_names = get_parts_objs(child, lev, root_to_load_file_name + '/' + root_to_load_file['name'])
            parts_objs = parts_objs + child_objs
            parts_names = parts_names + child_names 
    return parts_objs, parts_names


def get_norm_parts_points(parts_v,parts_f):
    max_size = 0
    parts_points=[]
    Rs = []
    ts = []
    sizes = []
    for v,f in zip(parts_v,parts_f):
        points = sample_pc(v,f)
        R, t, size = bbox(points)
        Rs.append(R)
        ts.append(t)
        sizes.append(size)
        points = np.dot(points -t, np.linalg.inv(R).transpose()) 
        parts_points.append(points)
        if max_size < np.max(size): max_size = np.max(size)
    parts_points = parts_points / max_size
    ts = ts / max_size

    return parts_points, np.array(Rs), np.array(ts), sizes

def merge_objs(vs,fs):
    
    newv = []
    newf = []
    num = 0

    for i in range(len(vs)):
        fs[i] += num
        num += len(vs[i])
    
    newv = np.concatenate(vs,axis=0)
    newf = np.concatenate(fs,axis=0)
    return newv,newf

def bbox(points):
    from sklearn.decomposition import PCA
    try: 
        to_origin, size = trimesh.bounds.oriented_bounds(obj=points, angle_digits=1)
        center = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
    
        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
        zdir = to_origin[2, :3]
    except:
       points = np.array(points)
       pca = PCA()
       pca.fit(points)
       pcomps = pca.components_
       points_local = np.matmul(pcomps, points.transpose()).transpose()
       all_max = points_local.max(axis=0)
       all_min = points_local.min(axis=0)
       center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
       size = all_max - all_min
       xdir = pcomps[0, :]
       xdir /= np.linalg.norm(xdir)
       ydir = pcomps[1, :]
       ydir /= np.linalg.norm(ydir)
       zdir = np.cross(xdir, ydir)
       zdir /= np.linalg.norm(zdir)
    
    R = np.vstack([xdir, ydir, zdir]).transpose().astype(np.float32).tolist()
    t = center.astype(np.float32).tolist()
    
    
    return R, t, size

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f
# num = 0
def sample_pc(v, f, n_points=1000):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points

def export_ply(out, v):
    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex '+str(v.shape[0])+'\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('property uchar red\n');
        fout.write('property uchar green\n');
        fout.write('property uchar blue\n');
        fout.write('end_header\n');
        # ipdb.set_trace()
        for i in range(v.shape[0]):
            cur_color=np.random.rand(3)
            cur_color=[1,1,1]
            # ipdb.set_trace()
            fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], \
                    int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)))
def is_same_geo(size1,size2):

    mean_bs = (np.linalg.norm(size1) + np.linalg.norm(size2)) / 2
    bs_diff = np.linalg.norm(size1 - size2)
    error = bs_diff / mean_bs
    
    box_thre = 0.1
    if error < box_thre:
        return True
    return False

def get_geo_part_ids(part_sizes,part_ids):
    class_ids = list(set(part_ids))
    geo_ind = 0
    geo_part_ids = [0 for i in range(len(part_ids))]
    for class_ind in class_ids:

        list_of_this_class = []
        for ind,part_id in enumerate(part_ids):
            if part_id == class_ind:
                list_of_this_class.append(ind)
        #print("list_of_this_class",list_of_this_class)
        for ind,part_ind in enumerate(list_of_this_class):
            flag = 0
            for pre_part_ind in list_of_this_class[:ind]:
                if is_same_geo(part_sizes[pre_part_ind],part_sizes[part_ind]):
                    flag = 1
                    geo_part_ids[part_ind] = geo_part_ids[pre_part_ind]
                    break
            if flag == 0:
                geo_part_ids[part_ind] = geo_ind
                geo_ind += 1
    return geo_part_ids


if __name__=="__main__":

    root_to_load_file = "../data/partnet_dataset/"
    root_to_save_file = "../prepare_data/shape_data/"
    # root_to_save_flle = "./prepared_data"
    cat_name = 'Lamp'
    cat_name = "Cabinet"
    cat_name = "Table"
    modes = ["train", "val", "test"]
    levels = [3,1,2]
    
    # import hier imformation
    fn_hier = root_to_load_file + "stats/after_merging_label_ids/" + cat_name + '-hier.txt'
    with open(fn_hier) as f:
        hier = f.readlines()
        hier = {'/'+s.split(' ')[1].replace('\n', ''):int(s.split(' ')[0]) for s in hier}

        print(hier)
    # for each level
    for level in levels:

        # import level information
        fn_level = root_to_load_file + "stats/after_merging_label_ids/" + cat_name + '-level-' + str(level) + ".txt"
        lev = [] 
        with open(fn_level) as f:
            lev = f.readlines()
            lev = ['/'+s.split(' ')[1].replace('\n', '') for s in lev]

        # for each mode 
        num = 0
        for mode in modes:
            
            #get the object list to deal with
            object_json =json.load(open(root_to_load_file + "/train_val_test_split/" + cat_name +"." + mode + ".json"))
            object_list = [int(object_json[i]['anno_id']) for i in range(len(object_json))]

            #for each object:
            for i,fn in enumerate(object_list):
                print("level ", level, " mode ", mode, " " ,fn," is start to convert!",i,"/",len(object_list))

                # get information in obj file
                parts_pcs, Rs, ts, parts_names, sizes = get_shape_info(fn, lev)
                

                # get class index and geo class index
                parts_ids = [hier[name] for name in parts_names]
                geo_part_ids = get_geo_part_ids(sizes, parts_ids)
                
                # gen sym_stick info
                sym = get_sym(parts_pcs)
                # get part poses from R , T
                parts_poses = []
                for R, t in zip(Rs, ts):
                    if np.linalg.det(R) < 0:
                        R = -R 
                    q = Quaternion(matrix=R)
                    q = np.array([q[i] for i in range(4)])
                    parts_pose = np.concatenate((t,q),axis=0)
                    parts_poses.append(parts_pose)
                parts_poses = np.array(parts_poses)
                new_dict = {v : k for k, v in hier.items()}
                dic_to_save = {"part_pcs":parts_pcs,"part_poses":parts_poses, "part_ids":parts_ids,"geo_part_ids":geo_part_ids,"sym":sym}
                np.save(root_to_save_file + str(fn)+"_level" + str(level) + ".npy", dic_to_save)






