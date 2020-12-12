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
