# dataset distribution is specified by a distribution file containing samples
import bpy
import math
import sys
import os
print(sys.path)
import numpy as np
import random
import struct
from numpy.linalg import inv
from math import *
import mathutils

def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    print(sys.path)
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def makeMaterial(name):
    mat = bpy.data.materials.new(name)
    mat.subsurface_scattering.use = True
    return mat

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def importParamBIN(origin_list, lookat_list, upvec_list):
	paramRotList = list()
	paramTransList = list()
	cutList = list()
	
	x0 = -10000
	y0 = -10000
	x1 = 10000
	y1 = 10000
	
	origin = np.array([eval(i) for i in origin_list.split(',')])
	lookat = np.array([eval(i) for i in lookat_list.split(',')])
	viewUp = np.array([eval(i) for i in upvec_list.split(',')])
	
	viewDir = origin - lookat
	viewDir = viewDir / np.linalg.norm(viewDir)
	viewRight = np.cross(viewUp, viewDir)
	viewRight= viewRight / np.linalg.norm(viewRight)
	viewUp = np.cross(viewDir, viewRight)
	viewUp = viewUp / np.linalg.norm(viewUp)
	
	R = np.ndarray((3, 3))
	R[0, 0] = viewRight[0]
	R[1, 0] = viewRight[1]
	R[2, 0] = viewRight[2]
	R[0, 1] = viewUp[0]
	R[1, 1] = viewUp[1]
	R[2, 1] = viewUp[2]
	R[0, 2] = viewDir[0]
	R[1, 2] = viewDir[1]
	R[2, 2] = viewDir[2]
	R = inv(R);
        
	paramRotList.append(R)
	
	T = np.ndarray((3, 1))
	T[0, 0] = origin[0]
	T[1, 0] = origin[1]
	T[2, 0] = origin[2]
	T = np.dot(-R, T)
	
	paramTransList.append(T)
	
	cutList.append([x0, y0, x1, y1]);
	
	return (paramRotList, paramTransList, cutList)

"""---------- main -----------"""
modelPath = sys.argv[6]
outPath = sys.argv[7]

print(sys.path)
modelId = os.path.basename(modelPath)[:-4]

bpy.ops.import_scene.obj(filepath=modelPath) 

bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
#bpy.context.scene.render.use_shadows = False
bpy.context.scene.render.use_raytrace = True
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = 0.2
bpy.context.scene.render.use_freestyle = False
bpy.context.scene.render.line_thickness = 0
bpy.context.scene.render.edge_threshold = 0
bpy.context.scene.render.edge_color = (1, 1, 1)
bpy.context.scene.render.use_edge_enhance = False

#bpy.context.mesh.show_normal_vertex = True;


# YOUR CODE START HERE

# fix mesh
scene = bpy.context.scene
for obj in scene.objects:
	if obj.type == 'MESH':
		scene.objects.active = obj
		bpy.ops.object.mode_set(mode='EDIT', toggle=False)
		bpy.ops.mesh.reveal()
		bpy.ops.mesh.select_all(action='SELECT')
		bpy.ops.mesh.normals_make_consistent()
		bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

# clear default lights
bpy.ops.object.select_by_type(type='LAMP')
bpy.ops.object.delete(use_global=False)

# set area lights
light_azimuth_deg = 0
light_elevation_deg  = 90
lightDist = 10
lx, ly, lz = obj_centened_camera_pos(lightDist, light_azimuth_deg, light_elevation_deg)
bpy.ops.object.lamp_add(type='AREA', view_align = False, location=(lx, ly, lz))
data = bpy.data.objects['Area'].data
data.energy = 1
data.distance = 5
#data.shape = 'SQUARE'
#data.shadow_ray_samples_x = 8

light_azimuth_deg = 0
light_elevation_deg  = 45
lightDist = 10
lx, ly, lz = obj_centened_camera_pos(lightDist, light_azimuth_deg, light_elevation_deg)
bpy.ops.object.lamp_add(type='AREA', view_align = False, location=(lx, ly, lz))
data = bpy.data.objects['Area.001'].data
data.energy = 1
data.distance = 5

#camObj.rotation_mode = 'XYZ'
#camObj.rotation_euler[0] = 0
#camObj.rotation_euler[1] = 0
#camObj.rotation_euler[2] = 0

outFileView = outPath;
bpy.data.objects['Area'].data.energy = 1;
bpy.data.objects['Area.001'].data.energy = 1;
bpy.context.scene.world.light_settings.environment_energy = 0.2
bpy.data.scenes['Scene'].render.filepath = outFileView;
bpy.ops.render.render( write_still=True )

