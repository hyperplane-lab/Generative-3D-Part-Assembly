import sys
import numpy as np

in_fn = sys.argv[1]
out_fn = sys.argv[2]

from render_using_blender import render_part_pts

pts = np.load(in_fn)
print(pts.shape)

render_part_pts(out_fn, pts, blender_fn='object_centered.blend')
