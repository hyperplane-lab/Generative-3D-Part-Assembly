import os
import torch
import numpy as np
from subprocess import call
from geometry_utils import load_obj, export_obj
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from quaternion import qrot
from colors import colors




cube_mesh = load_obj(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cube.obj'), no_normal=True)
cube_v_torch = torch.from_numpy(cube_mesh['vertices'])
cube_v = cube_mesh['vertices'] / 100
cube_f = cube_mesh['faces']

"""
    pts: N x 3
"""
def render_pts(out_fn, pts, blender_fn='blank.blend'):
    all_v = [np.zeros((0, 3), dtype=np.float32)]; 
    all_f = [np.zeros((0, 3), dtype=np.int32)];
    for i in range(pts.shape[0]):
        all_v.append(cube_v + pts[i])
        all_f.append(cube_f + 8 * i)
    all_v = np.vstack(all_v)
    all_f = np.vstack(all_f)
    with open(out_fn+'.obj', 'w') as fout:
        fout.write('mtllib %s\n' % (out_fn.split('/')[-1]+'.mtl'))
        for i in range(all_v.shape[0]):
            fout.write('v %f %f %f\n' % (all_v[i, 0], all_v[i, 1], all_v[i, 2]))
        fout.write('usemtl f0\n')
        for i in range(all_f.shape[0]):
            fout.write('f %d %d %d\n' % (all_f[i, 0], all_f[i, 1], all_f[i, 2]))
    with open(out_fn+'.mtl', 'w') as fout:
        fout.write('newmtl f0\nKd 1 0 0\n')
    cmd = 'cd %s && blender -noaudio --background %s --python render_blender.py %s %s > /dev/null' \
            % (os.path.join(os.path.dirname(os.path.abspath(__file__))), \
            blender_fn, out_fn+'.obj', out_fn)
    call(cmd, shell=True)

"""
    pts: P x N x 3 (P <= 20)
"""
def render_part_pts(out_fn, pts, blender_fn='blank.blend'):
    fobj = open(out_fn+'.obj', 'w')
    fobj.write('mtllib %s\n' % (out_fn.split('/')[-1]+'.mtl'))
    fmtl = open(out_fn+'.mtl', 'w')
    num_part = pts.shape[0]
    num_point = pts.shape[1]
    for pid in range(num_part):
        all_v = [np.zeros((0, 3), dtype=np.float32)]; 
        all_f = [np.zeros((0, 3), dtype=np.int32)];
        for i in range(num_point):
            all_v.append(cube_v + pts[pid, i])
            all_f.append(cube_f + 8 * (pid*num_point+i))
        all_v = np.vstack(all_v)
        all_f = np.vstack(all_f)
        for i in range(all_v.shape[0]):
            fobj.write('v %f %f %f\n' % (all_v[i, 0], all_v[i, 1], all_v[i, 2]))
        fobj.write('usemtl f%d\n' % pid)
        for i in range(all_f.shape[0]):
            fobj.write('f %d %d %d\n' % (all_f[i, 0], all_f[i, 1], all_f[i, 2]))
        fmtl.write('newmtl f%d\nKd %f %f %f\n' % (pid, colors[pid][0], colors[pid][1], colors[pid][2]))
    fobj.close()
    fmtl.close()
    cmd = 'cd %s && blender -noaudio --background %s --python render_blender.py %s %s > /dev/null' \
            % (os.path.join(os.path.dirname(os.path.abspath(__file__))), \
            blender_fn, out_fn+'.obj', out_fn)
    call(cmd, shell=True)
"""
    relation: P x P
    select_part_ind: 1
    pts: P x N x 3 (P <= 20)
"""
def render_part_pts_relation(out_fn, pts, select_pair_ind, relation, blender_fn='blank.blend'):
    fobj = open(out_fn+'.obj', 'w')
    fobj.write('mtllib %s\n' % (out_fn.split('/')[-1]+'.mtl'))
    fmtl = open(out_fn+'.mtl', 'w')
    num_part = pts.shape[0]
    num_point = pts.shape[1]
    # from matplotlib import colors
    from matplotlib import cm
    for pid in range(num_part):
        all_v = [np.zeros((0, 3), dtype=np.float32)]; 
        all_f = [np.zeros((0, 3), dtype=np.int32)];
        for i in range(num_point):
            all_v.append(cube_v + pts[pid, i])
            all_f.append(cube_f + 8 * (pid*num_point+i))
        all_v = np.vstack(all_v)
        all_f = np.vstack(all_f)
        for i in range(all_v.shape[0]):
            fobj.write('v %f %f %f\n' % (all_v[i, 0], all_v[i, 1], all_v[i, 2]))
        fobj.write('usemtl f%d\n' % pid)
        for i in range(all_f.shape[0]):
            fobj.write('f %d %d %d\n' % (all_f[i, 0], all_f[i, 1], all_f[i, 2]))
        if pid == select_pair_ind:
            cmap = cm.get_cmap('winter')
            c = cmap(1.00) # 会根据 score map 的值从 cmap 中找 color，返回的是 rgba 图像
            fmtl.write('newmtl f%d\nKd %f %f %f\n' % (pid, 1, 0, 0))
        else:
            print("relation between",select_pair_ind," and ", pid ,": ",relation[select_pair_ind][pid].item())
            cmap = cm.get_cmap('winter')
            c = cmap(relation[select_pair_ind][pid].item()) # 会根据 score map 的值从 cmap 中找 color，返回的是 rgba 图像
            fmtl.write('newmtl f%d\nKd %f %f %f\n' % (pid, c[0], c[1], c[2]))

    fobj.close()
    fmtl.close()
    cmd = 'cd %s && blender -noaudio --background %s --python render_blender.py %s %s > /dev/null' \
            % (os.path.join(os.path.dirname(os.path.abspath(__file__))), \
            blender_fn, out_fn+'.obj', out_fn)
    call(cmd, shell=True)

"""
    专门用来标出：被检查是否连接的pair的问题
    pts: P x N x 3 (P <= 20)
    out_fn: 输出地址
    blender_fn:
    select_pair_index:被渲染的part的编号，是一个list，默认长度是2.
"""
def render_part_pts_connect_points(out_fn, pts, select_pair_index, blender_fn='blank.blend'):
    fobj = open(out_fn+'.obj', 'w')
    fobj.write('mtllib %s\n' % (out_fn.split('/')[-1]+'.mtl'))
    fmtl = open(out_fn+'.mtl', 'w')
    num_part = pts.shape[0]
    num_point = pts.shape[1]
    for pid in range(num_part):
        all_v = [np.zeros((0, 3), dtype=np.float32)]; 
        all_f = [np.zeros((0, 3), dtype=np.int32)];
        for i in range(num_point):
            all_v.append(cube_v + pts[pid, i])
            all_f.append(cube_f + 8 * (pid*num_point+i))
        all_v = np.vstack(all_v)
        all_f = np.vstack(all_f)
        for i in range(all_v.shape[0]):
            fobj.write('v %f %f %f\n' % (all_v[i, 0], all_v[i, 1], all_v[i, 2]))
        fobj.write('usemtl f%d\n' % pid)
        for i in range(all_f.shape[0]):
            fobj.write('f %d %d %d\n' % (all_f[i, 0], all_f[i, 1], all_f[i, 2]))
        if pid in select_pair_index:
            color_ind = 1
        else:
            color_ind = 2
        fmtl.write('newmtl f%d\nKd %f %f %f\n' % (pid, colors[color_ind][0], colors[color_ind][1], colors[color_ind][2]))
    fobj.close()
    fmtl.close()
    cmd = 'cd %s && blender -noaudio --background %s --python render_blender.py %s %s > /dev/null' \
            % (os.path.join(os.path.dirname(os.path.abspath(__file__))), \
            blender_fn, out_fn+'.obj', out_fn)
    call(cmd, shell=True)
