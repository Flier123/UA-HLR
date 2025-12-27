import os

from os.path import join as ospj
from os.path import expanduser
# from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from src.models.networks import BayesCap_MLP

### training and evaluation
def emb_mae(x1, x2):
    m = torch.abs(x1-x2).mean()
    return m

def emb_mse(x1, x2):
    m = torch.pow(torch.abs(x1-x2),2).mean()
    return m
# 数据不确定性
def get_GGuncer(x_alpha, x_beta, c1=3, c2=2.8):
    a = 1/(x_alpha + 1e-5)
    a = torch.clip(a, min=1e-4, max=5)
    b = x_beta + 0.1
    b = torch.clip(b, min=0.1, max=5)
    u = (a**2)*torch.exp(torch.lgamma(3/b))/torch.exp(torch.lgamma(1.0/b))
    return u

class GenGaussLoss(nn.Module):
	def __init__(
		self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
	) -> None:
		super(GenGaussLoss, self).__init__()
		self.reduction = reduction
		self.alpha_eps = alpha_eps
		self.beta_eps = beta_eps
		self.resi_min = resi_min
		self.resi_max = resi_max
	
	def forward(
		self, 
		mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor
	):
		one_over_alpha1 = one_over_alpha + self.alpha_eps
		beta1 = beta + self.beta_eps

		resi = torch.abs(mean - target)
		# resi = torch.pow(resi*one_over_alpha1, beta1).clamp(min=self.resi_min, max=self.resi_max)
		resi = (resi*one_over_alpha1*beta1).clamp(min=self.resi_min, max=self.resi_max)
		## check if resi has nans
		if torch.sum(resi != resi) > 0:
			print('resi has nans!!')
			return None
		
		log_one_over_alpha = torch.log(one_over_alpha1)
		log_beta = torch.log(beta1)
		lgamma_beta = torch.lgamma(torch.pow(beta1, -1))
		
		if torch.sum(log_one_over_alpha != log_one_over_alpha) > 0:
			print('log_one_over_alpha has nan')
		if torch.sum(lgamma_beta != lgamma_beta) > 0:
			print('lgamma_beta has nan')
		if torch.sum(log_beta != log_beta) > 0:
			print('log_beta has nan')
		
		l = resi - log_one_over_alpha + lgamma_beta - log_beta

		if self.reduction == 'mean':
			return l.mean()
		elif self.reduction == 'sum':
			return l.sum()
		else:
			print('Reduction not supported')
			return None

class TempCombLoss(nn.Module):
	def __init__(
		self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
	) -> None:
		super(TempCombLoss, self).__init__()
		self.reduction = reduction
		self.alpha_eps = alpha_eps
		self.beta_eps = beta_eps
		self.resi_min = resi_min
		self.resi_max = resi_max

		self.L_GenGauss = GenGaussLoss(
			reduction=self.reduction,
			alpha_eps=self.alpha_eps, beta_eps=self.beta_eps, 
			resi_min=self.resi_min, resi_max=self.resi_max
		)
		self.L_l1 = nn.L1Loss(reduction=self.reduction)
	
	def forward(
		self,
		mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor,
		T1: float, T2: float
	):
		l1 = self.L_l1(mean, target)
		l2 = self.L_GenGauss(mean, one_over_alpha, beta, target)
		l = T1*l1 + T2*l2

		return l



import os
from tqdm import tqdm

def train_PEM(
        moco_model,            # 冻结的 moco_model 模型
        train_loader,        # 训练集 DataLoader
        val_loader,          # 验证集 DataLoader
        Cri=TempCombLoss(),  # 默认损失函数
        device='cuda',
        init_lr=5e-4,
        num_epochs=10,
        ckpt_path=None,     # 保存权重前缀
        T1=1e0, T2=5e-2,                  # 损失函数温度系数
        epoch_index=None,writer=None

):
    # ---- 1. 模型准备 ----
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_features,_ = moco_model(dummy_input,cls_only=True)
        moco_dim = dummy_features.shape[-1]
    # print(f"图像特征维度: {moco_dim}")

    # 创建概率适配器
    BayesCap_Net = BayesCap_MLP(
        inp_dim=moco_dim,
        out_dim=moco_dim,
        hid_dim=moco_dim//2,
        num_layers=3
    ).to(device)

    # ---- 2. 优化器与调度器 ----
    optimizer = torch.optim.Adam(
        list(BayesCap_Net.parameters()),
        lr=init_lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # ---- 2.1 断点续训检查 ----
    ckpt_file = ckpt_path
    all_loss = []
    best_val_metric = float("inf")  # 用于保存最优（越小越好）
    start_epoch = 0

    if os.path.exists(ckpt_file):
        print(f"发现已有检查点 {ckpt_file}，正在加载...")
        checkpoint = torch.load(ckpt_file, map_location=device)
        BayesCap_Net.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # optim_scheduler.load_state_dict(checkpoint["scheduler"])
        # start_epoch = checkpoint["epoch"] + 1
        # best_val_metric = checkpoint["best_val_metric"]
        # print(f"从第 {start_epoch} 个 epoch 继续训练 (当前 best_val_metric={best_val_metric:.6f})")
    else:
        print("未发现已有检查点，将从头开始训练。")

    # 要先训练好probVLM
    # if epoch_index == 0:
    #       num_epochs = 5


    # ---- 3. 主循环 ----
    for epoch in range(start_epoch,num_epochs):
        epoch_loss = 0.0
        BayesCap_Net.train()

        with tqdm(train_loader, unit='batch') as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f'Epoch {epoch}/{num_epochs}')

                # ---- 4. 数据准备 ----
                xI = batch[0][1].to(device)
                with torch.no_grad():
                    xfI,_ = moco_model(xI,cls_only=True)
                    # xfI = torch.nn.functional.layer_norm(xfI, xfI.shape[1:])

                # ---- 5. 前向计算 ----
                img_mu, img_1alpha, img_beta = BayesCap_Net(xfI)

                # ---- 6. 损失计算 ----
                optimizer.zero_grad()
                loss = Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
                loss.backward()
                optimizer.step()
                
                if writer:
                    writer.add_scalar('loss/probVLM_batch', loss.item(), epoch*len(train_loader)+idx)
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        epoch_loss /= len(train_loader)
        if writer:
            writer.add_scalar('loss/probVLM', epoch_loss, epoch)
        all_loss.append(epoch_loss)
        print(f'Epoch {epoch}/{num_epochs} | Avg. loss: {epoch_loss:.6f}')

        # ---- 7. 验证并保存检查点 ----
        val_metric = eval_PEM(moco_model, BayesCap_Net, val_loader, device)

        checkpoint = {
            "model": BayesCap_Net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": optim_scheduler.state_dict(),
            "epoch": epoch,
            "val_metric": val_metric,
            "best_val_metric": best_val_metric
        }
        # 总是保存最后一次
        if ckpt_file:
            torch.save(checkpoint, ckpt_file)
        root, ext = os.path.splitext(ckpt_file)      # ("../probVLM_ckpt/ProbVLM_xxx_xxx_xxx", ".pth")
        best_p = root + "_best" + ext         # "../probVLM_ckpt/ProbVLM_xxx_xxx_xxx_best.pth"

        # 仅在验证指标更优时保存最优
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(checkpoint, best_p)
            print(f"保存最优模型 (val_metric={best_val_metric:.6f}) 到 {best_p}")
        optim_scheduler.step()

    print("结束train_PEM训练!")

def eval_PEM(moco_model, BayesCap_Net, eval_loader, device='cuda'):
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    mean_mse = 0.0
    mean_mae = 0.0
    num_imgs = 0

    with tqdm(eval_loader, unit='batch') as tepoch:
        for idx, batch in enumerate(tepoch):
            tepoch.set_description('Validating ...')
            xI = batch[0].to(device)

            with torch.no_grad():
                xfI,_ = moco_model(xI,cls_only=True)          # 冻结  提取嵌入
                # xfI = torch.nn.functional.layer_norm(xfI, xfI.shape[1:])
                img_mu, img_1alpha, img_beta = BayesCap_Net(xfI)

                # 逐样本计算 MSE/MAE（均值与真值差异）
                n_batch = img_mu.shape[0]
                for j in range(n_batch):
                    num_imgs += 1
                    mean_mse += emb_mse(img_mu[j], xfI[j]) 
                    mean_mae += emb_mae(img_mu[j], xfI[j]) 

    mean_mse /= num_imgs
    mean_mae /= num_imgs
    print(f'Avg. MSE: {mean_mse} | Avg. MAE: {mean_mae}')
    return mean_mae          # 返回 MAE 作为评价指标


import torch
import os

def extract_PEM_features(
    moco_model, 
    images, 
    ckpt_path="../ProbVLM.pth", 
    device='cuda',
    hid_dim=256, num_layers=3, p_drop=0.05
):
    """
    提取 ProbVLM 输出的均值 μ 和分布参数 (1/α, β)

    Args:
        CLIP_Net: 已冻结好的 CLIP 模型
        images: 输入图像 batch (tensor, shape: [B, 3, H, W])
        ckpt_path: 训练好的 ProbVLM 权重路径 (ProbVLM.pth)
        device: 运行设备 (默认 'cuda')
        hid_dim, num_layers, p_drop: 构建 BayesCap_MLP 的超参数 (需与训练时一致)

    Returns:
        img_mu: 预测的均值向量 (B, D)
        img_1alpha: 预测的尺度倒数 (B, D)
        img_beta: 预测的形状参数 (B, D)
    """
    # ---- 1. 获取 CLIP 特征维度 ----
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_features,_ = moco_model(dummy_input,cls_only=True)
        moco_dim = dummy_features.shape[-1]
    # print(f"图像特征维度: {moco_dim}")

    # ---- 2. 构建 BayesCap_Net ----
    BayesCap_Net = BayesCap_MLP(
        inp_dim=moco_dim,
        out_dim=moco_dim,
        hid_dim=moco_dim//2,
        num_layers=num_layers,
    ).to(device)

    # ---- 3. 加载训练好的权重 ----
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到权重文件: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model" in checkpoint:   # 兼容断点续训保存格式
        BayesCap_Net.load_state_dict(checkpoint["model"])
    else:                       # 兼容只保存 state_dict 的情况
        BayesCap_Net.load_state_dict(checkpoint)
    BayesCap_Net.eval()

    # ---- 4. 前向推理 ----
    with torch.no_grad():
        images = images.to(device)
        xfI,_ = moco_model(images,cls_only=True)                  # CLIP embedding
        # xfI = torch.nn.functional.layer_norm(xfI, xfI.shape[1:])
        img_mu, img_1alpha, img_beta = BayesCap_Net(xfI)     # ProbVLM 输出参数

    return img_mu, img_1alpha, img_beta
