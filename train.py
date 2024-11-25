#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

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

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tinycudann_test/scripts")
sys.path.insert(0, SCRIPTS_DIR)

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class Image(torch.nn.Module):
	def __init__(self, filename, device):
		super(Image, self).__init__()
		self.data = read_image(filename)
		self.shape = self.data.shape
		self.data = torch.from_numpy(self.data).float().to(device)

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
			indices = xs.long()
			lerp_weights = xs - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			return (
				self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("ply", nargs="?", default="data/lego.ply", help="Image to match")
	parser.add_argument("config", nargs="?", default="data/config_hash_own.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
	parser.add_argument("result_filename", nargs="?", default="", help="Number of training steps")

	args = parser.parse_args()
	return args

def contract_to_unisphere(
        x: torch.Tensor,
        aabb: torch.Tensor=torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'),
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        # x缩放到aabb盒子的归一化，0为中点，-1 1为box边界
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        # 超过盒子的（球）
        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            # 最终把整个世界缩放到0-1
            return x

def cal_sh_color(dir_pp,f_dc,f_rest):
	f_dc = torch.tensor(f_dc).cuda()
	cal_0 = f_dc * 0.28209479177387814
	f_rest = torch.tensor(f_rest).cuda()
	
	sh1_1 = torch.cat((f_rest[:,0].unsqueeze(1),f_rest[:,15].unsqueeze(1),f_rest[:,30].unsqueeze(1)),dim = 1)
	sh1_2 = torch.cat((f_rest[:,1].unsqueeze(1),f_rest[:,16].unsqueeze(1),f_rest[:,31].unsqueeze(1)),dim = 1)
	sh1_3 = torch.cat((f_rest[:,2].unsqueeze(1),f_rest[:,17].unsqueeze(1),f_rest[:,32].unsqueeze(1)),dim = 1)
	
	sh2_1 = torch.cat((f_rest[:,3].unsqueeze(1),f_rest[:,18].unsqueeze(1),f_rest[:,33].unsqueeze(1)),dim = 1)
	sh2_2 = torch.cat((f_rest[:,4].unsqueeze(1),f_rest[:,19].unsqueeze(1),f_rest[:,34].unsqueeze(1)),dim = 1)
	sh2_3 = torch.cat((f_rest[:,5].unsqueeze(1),f_rest[:,20].unsqueeze(1),f_rest[:,35].unsqueeze(1)),dim = 1)
	sh2_4 = torch.cat((f_rest[:,6].unsqueeze(1),f_rest[:,21].unsqueeze(1),f_rest[:,36].unsqueeze(1)),dim = 1)
	sh2_5 = torch.cat((f_rest[:,7].unsqueeze(1),f_rest[:,22].unsqueeze(1),f_rest[:,37].unsqueeze(1)),dim = 1)

	sh3_1 = torch.cat((f_rest[:,8].unsqueeze(1),f_rest[:,23].unsqueeze(1),f_rest[:,38].unsqueeze(1)),dim = 1)
	sh3_2 = torch.cat((f_rest[:,9].unsqueeze(1),f_rest[:,24].unsqueeze(1),f_rest[:,39].unsqueeze(1)),dim = 1)
	sh3_3 = torch.cat((f_rest[:,10].unsqueeze(1),f_rest[:,25].unsqueeze(1),f_rest[:,40].unsqueeze(1)),dim = 1)
	sh3_4 = torch.cat((f_rest[:,11].unsqueeze(1),f_rest[:,26].unsqueeze(1),f_rest[:,41].unsqueeze(1)),dim = 1)
	sh3_5 = torch.cat((f_rest[:,12].unsqueeze(1),f_rest[:,27].unsqueeze(1),f_rest[:,42].unsqueeze(1)),dim = 1)
	sh3_6 = torch.cat((f_rest[:,13].unsqueeze(1),f_rest[:,28].unsqueeze(1),f_rest[:,43].unsqueeze(1)),dim = 1)
	sh3_7 = torch.cat((f_rest[:,14].unsqueeze(1),f_rest[:,29].unsqueeze(1),f_rest[:,44].unsqueeze(1)),dim = 1)




	# sh1_1 ,sh1_2,sh1_3= f_rest[:,:3],f_rest[:,3:6],f_rest[:,6:9]
	SH_C1 = 0.4886025119029
	cal_1 = -sh1_1 * dir_pp[:,1].unsqueeze(1) * SH_C1 + sh1_2 * dir_pp[:,2].unsqueeze(1) * SH_C1 - sh1_3 * dir_pp[:,0].unsqueeze(1) * SH_C1
	x = dir_pp[:,0].unsqueeze(1)
	y = dir_pp[:,1].unsqueeze(1)
	z = dir_pp[:,2].unsqueeze(1)

	xx = dir_pp[:,0].unsqueeze(1) * dir_pp[:,0].unsqueeze(1)
	yy = dir_pp[:,1].unsqueeze(1) * dir_pp[:,1].unsqueeze(1)
	zz = dir_pp[:,2].unsqueeze(1) * dir_pp[:,2].unsqueeze(1)
	xy = dir_pp[:,0].unsqueeze(1) * dir_pp[:,1].unsqueeze(1)
	xz = dir_pp[:,0].unsqueeze(1) * dir_pp[:,2].unsqueeze(1)
	yz = dir_pp[:,1].unsqueeze(1) * dir_pp[:,2].unsqueeze(1)

	SH_C2 = [1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396]
	cal_2 = SH_C2[0] * xy * sh2_1 + SH_C2[1] * yz * sh2_2 + SH_C2[2] * (2.0 * zz - xx - yy) * sh2_3 + SH_C2[3] * xz * sh2_4 + SH_C2[4] * (xx-yy) * sh2_5
	SH_C3 = [-0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435]
	cal_3 = SH_C3[0] * y * (3.0 * xx - yy) * sh3_1 + SH_C3[1] * xy * z * sh3_2 + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh3_3 + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh3_4 + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh3_5 + SH_C3[5] * z * (xx - yy) * sh3_6 + SH_C3[6] * x * (xx - 3.0 * yy) * sh3_7
	colors = cal_0 + cal_1 + cal_2 + cal_3 + 0.5
	return colors

def cal_mlp_color(mlp_sh):
	# f_dc = torch.tensor(f_dc).cuda()
	colors = mlp_sh * 0.28209479177387814

	return colors + 0.5

def save_ply(path, mlp_sh,others):


	
	[xyz,normals,opacities,scale,rotation,f_dc,f_rest] = others
	xyz =xyz.detach().cpu().numpy()
	normals = np.zeros_like(xyz)
	
	f_dc = mlp_sh
	f_rest = np.zeros_like(f_rest)
	opacities = opacities
	scale = scale
	rotation = rotation


	l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
	for i in range(3):
		l.append('f_dc_{}'.format(i))
	for i in range(45):
		l.append('f_rest_{}'.format(i))
	l.append('opacity')
	for i in range(3):
		l.append('scale_{}'.format(i))
	for i in range(4):
		l.append('rot_{}'.format(i))

	dtype_full = [(attribute, 'f4') for attribute in l]

	elements = np.empty(xyz.shape[0], dtype=dtype_full)
	attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
	elements[:] = list(map(tuple, attributes))
	el = PlyElement.describe(elements, 'vertex')
	PlyData([el]).write(path)
	print("saved")


def save_ply_small(path, mlp_color,others):


	
	[xyz,normals,opacities,scale,rotation,f_dc,f_rest] = others
	xyz =xyz.detach().cpu().numpy()
	normals = np.zeros_like(xyz)
	
	f_dc = (mlp_color-0.5)/0.28209479177387814
	f_rest = np.zeros_like(f_rest)
	opacities = opacities
	scale = scale
	rotation = rotation


	l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
	# for i in range(3):
	# 	l.append('f_dc_{}'.format(i))
	# for i in range(45):
	# 	l.append('f_rest_{}'.format(i))
	l.append('opacity')
	for i in range(3):
		l.append('scale_{}'.format(i))
	for i in range(4):
		l.append('rot_{}'.format(i))

	dtype_full = [(attribute, 'f4') for attribute in l]
	
	elements = np.empty(xyz.shape[0], dtype=dtype_full)
	attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
	elements[:] = list(map(tuple, attributes))
	el = PlyElement.describe(elements, 'vertex')
	PlyData([el]).write(path)
	print("saved")


if __name__ == "__main__":
	print("================================================================")
	print("This script replicates the behavior of the native CUDA example  ")
	print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
	print("================================================================")

	print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

	device = torch.device("cuda")
	args = get_args()

	with open(args.config) as config_file:
		config = json.load(config_file)

	plydata = PlyData.read(args.ply)
	xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])), axis=1)
	f_dc  = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                np.asarray(plydata.elements[0]["f_dc_1"]),
                np.asarray(plydata.elements[0]["f_dc_2"])), axis=1)
	
	for i in range(45):
		name = "f_rest_" + str(i)
		f_re_now = np.asarray(plydata.elements[0][name])
		try:
			f_re =  np.vstack((f_re,f_re_now))
		except:
			f_re = np.array([np.asarray(plydata.elements[0][name])])
	f_rest = f_re.T
	normals = np.zeros_like(xyz)
	opacities = np.asarray(plydata.elements[0]["opacity"]).reshape(-1,1)
	scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
					np.asarray(plydata.elements[0]["scale_1"]),
					np.asarray(plydata.elements[0]["scale_2"])), axis=1)
	rotation = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
					np.asarray(plydata.elements[0]["rot_1"]),
					np.asarray(plydata.elements[0]["rot_2"]),
					np.asarray(plydata.elements[0]["rot_3"])), axis=1)
	


	# image = Image(args.image, device)
	recolor = tcnn.Encoding(
			n_input_dims=3,
			encoding_config={
			"otype": "HashGrid",
			"n_levels": 16,
			"n_features_per_level": 2,
			"log2_hashmap_size": 16,
			"base_resolution": 16,
			"per_level_scale": 1.447,
		},)
	direction_encoding = tcnn.Encoding(
		n_input_dims=3,
		encoding_config={
			"otype": "SphericalHarmonics",
			"degree": 3 
		},)
	mlp_head = tcnn.Network(
                n_input_dims=(direction_encoding.n_output_dims+recolor.n_output_dims),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },)
	
	other_params = []
	for params in recolor.parameters():
		other_params.append(params)
	for params in mlp_head.parameters():
		other_params.append(params)

	optimizer = torch.optim.Adam(other_params, lr=0.01, eps=1e-15)

	xyz = torch.tensor(xyz).cuda()
	
	# model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	# print(model)
	

	#===================================================================================================
	# The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
	# tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
	#===================================================================================================
	# encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
	# network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
	# model = torch.nn.Sequential(encoding, network)


	optimizer = torch.optim.Adam(other_params, lr=0.01, eps=1e-15)
	prev_time = time.perf_counter()

	for i in range(args.n_steps):
		xyz_unisphere = contract_to_unisphere(xyz.half())
		feature = recolor(xyz_unisphere)
		# dir_pp = torch.zeros_like(xyz)
		dir_pp = torch.rand(xyz.shape).cuda()
		dir_pp = 2 * dir_pp - 1
		dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
		
		mlp_sh = mlp_head(torch.cat([feature, direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
		mlp_color = cal_mlp_color(mlp_sh)
		sh_color = cal_sh_color(dir_pp,f_dc,f_rest).unsqueeze(1)

		loss = F.l1_loss(mlp_color,sh_color)


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		
		if i % 100 == 0:
			loss_val = loss.item()
			torch.cuda.synchronize()
			elapsed_time = time.perf_counter() - prev_time
			prev_time = time.perf_counter()
			print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")
			
			path = f"train_{i}.ply"
			path_small = f"train_small_{i}.ply"
			with torch.no_grad():
				others = [xyz,normals,opacities,scale,rotation,f_dc,f_rest]
				# mlp_color = mlp_head(torch.cat([feature, direction_encoding(torch.zeros_like(dir_pp))], dim=-1)).unsqueeze(1)
				save_ply("results/" + path, mlp_sh.squeeze().detach().cpu().numpy(),others)
				save_ply_small("results/" + path_small, mlp_sh.squeeze().detach().cpu().numpy(),others)
				pth_name = f"point_cloud_{i}.pth"
				torch.save(torch.nn.ModuleList([recolor, mlp_head]).state_dict(), os.path.join("results", pth_name)) 
		# 	
		# 	print(f"Writing '{path}'... ", end="")
		# 	with torch.no_grad():
		# 		write_image(path, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
		# 	print("done.")

		# 	# Ignore the time spent saving the image
			

		# 	if i > 0 and interval < 1000:
		# 		interval *= 10


	tcnn.free_temporary_memory()