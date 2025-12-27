import sys

import torch


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  # p_i_j = compute_joint(x_out, x_tf_out)

  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
  su_temp1 = x_out.unsqueeze(2)
  su_temp2 = x_tf_out.unsqueeze(1)
  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k #这两个相乘有什么用？
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise 为什么要对称
  p_i_j = p_i_j / p_i_j.sum()  # normalise




  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
  #                           - torch.log(p_j) \
  #                           - torch.log(p_i))

  # loss_no_lamb = loss_no_lamb.sum()

  return loss


import torch
import sys

def IID_loss_entropy_exp_weighted(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """
    IID Loss with teacher entropy-based exponential weighting.
    
    x_out: 学生模型输出，softmax, shape [B, K]
    x_tf_out: 教师模型输出，softmax, shape [B, K]
    lamb: IID loss hyperparameter
    返回: 标量 IID loss
    """
    B, K = x_out.size()

    # 1️⃣ 教师熵
    entropy = -torch.sum(x_tf_out * torch.log(x_tf_out + EPS), dim=1)  # [B]

    # 2️⃣ 最大熵
    max_entropy = torch.log(torch.tensor(K, dtype=entropy.dtype, device=entropy.device))  # ln(K)
    
    # 3️⃣ 权重: 熵越低权重越大
    weight = torch.exp(-entropy / max_entropy)  # [B]
    weight = weight.view(B, 1)  # [B,1]，用于广播

    # 4️⃣ 加权教师输出
    weighted_teacher = x_tf_out * weight  # [B, K]

    # 5️⃣ 构造联合概率矩阵
    p_i_j = x_out.unsqueeze(2) * weighted_teacher.unsqueeze(1)  # [B, K, K]
    p_i_j = p_i_j.sum(dim=0)  # [K, K]
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # 对称化
    p_i_j = p_i_j / (p_i_j.sum() + EPS)  # 归一化

    # 6️⃣ 边缘概率
    p_i = p_i_j.sum(dim=1).view(K, 1).expand(K, K)
    p_j = p_i_j.sum(dim=0).view(1, K).expand(K, K)

    # 7️⃣ IID loss
    loss_matrix = -p_i_j * (torch.log(p_i_j + EPS) - lamb * torch.log(p_j + EPS) - lamb * torch.log(p_i + EPS))
    loss = loss_matrix.sum()

    return loss



def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j