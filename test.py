import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
from plyfile import PlyData, PlyElement
import torch.nn.functional as F

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()
	
# ply_path = "/home/zhaoyibin/3DRE/3DGS/GS_compress/Compact-3DGS/output/lego_own/point_cloud/iteration_30000/point_cloud.ply"
# ply_path_2 = "/home/zhaoyibin/3DRE/3DGS/GS_compress/Compact-3DGS/output/lego_own/point_cloud/iteration_30000/point_cloud_own.ply"
# plydata = PlyData.read(ply_path)
# plydata_2 = PlyData.read(ply_path_2)

# print("end")
	
    