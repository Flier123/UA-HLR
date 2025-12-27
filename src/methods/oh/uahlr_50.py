import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
from datetime import datetime
from src.utils.PEM import *
import torch.nn.functional as F
from PIL import ImageFilter
from src.data.data_list import ImageList_idx_adacon, NCropsTransform,ImageList_idx_aug_fix
from PIL import ImageFilter  # 高斯模糊需要
from typing import Tuple


logger = logging.getLogger(__name__)

class AdaMoCo(nn.Module):
    def __init__(self, netF, netB, netC, momentum_netF, momentum_netB, momentum_netC, 
                 features_length, num_classes, dataset_length, temporal_length=5):
        super(AdaMoCo, self).__init__()

        self.m = 0.999  # momentum coefficient
        self.first_update = True

        # Query networks (trainable)
        self.netF = netF
        self.netB = netB  
        self.netC = netC
        
        # Key networks (momentum updated)
        self.momentum_netF = momentum_netF
        self.momentum_netB = momentum_netB
        self.momentum_netC = momentum_netC
        
        # Freeze momentum networks
        self.momentum_netF.requires_grad_(False)
        self.momentum_netB.requires_grad_(False)
        self.momentum_netC.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0
        self.T_moco = 0.07  # temperature for contrastive learning

        # Queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        # Memory buffers
        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer("labels", torch.randint(0, num_classes, (self.K,)))
        self.register_buffer("idxs", torch.randint(0, dataset_length, (self.K,)))
        self.register_buffer("mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length)))
        self.register_buffer("real_labels", torch.randint(0, num_classes, (dataset_length,)))

        self.features = F.normalize(self.features, dim=0)
        
        # Move to GPU
        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder networks"""
        # Update netF
        for param_q, param_k in zip(self.netF.parameters(), self.momentum_netF.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        
        # Update netB
        for param_q, param_k in zip(self.netB.parameters(), self.momentum_netB.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        
        # Update netC
        for param_q, param_k in zip(self.netC.parameters(), self.momentum_netC.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        """Update memory queue and temporal labels"""
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label
        self.queue_ptr = end % self.K

        # Update temporal memory
        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        """Get memory features and labels"""
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Forward pass for AdaMoCo
        Args:
            im_q: query images (weak augmentation)
            im_k: key images (strong augmentation), optional
            cls_only: if True, only return classification features and logits
        """
        # Query path: netF -> netB -> netC
        feats_q = self.netB(self.netF(im_q))  # bottleneck features
        logits_q = self.netC(feats_q)  # classification logits

        if cls_only:
            return feats_q, logits_q

        # Normalize query features for contrastive learning
        q = F.normalize(feats_q, dim=1)

        # Key path (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update momentum networks
            
            k_feats = self.momentum_netB(self.momentum_netF(im_k))  # key features
            k = F.normalize(k_feats, dim=1)  # normalized key features

        # Compute contrastive logits
        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # Negative logits: NxK  
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])

        # Concatenate positive and negative logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits_ins /= self.T_moco

        return feats_q, logits_q, logits_ins, k

class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

class ResNetDomainNet126(torch.nn.Module):
    """
    Architecture used for DomainNet-126
    """
    def __init__(self, arch="resnet50", checkpoint_path=None, num_classes=126, bottleneck_dim=256):
        super().__init__()

        self.arch = arch
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm_dim = 0

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = models.__dict__[self.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottleneck (last fc as bottleneck)
        else:
            model = models.__dict__[self.arch](pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, self.bottleneck_dim)
            bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = torch.nn.Sequential(model, bn)
            self._output_dim = self.bottleneck_dim

        self.fc = torch.nn.Linear(self.output_dim, num_classes)

        if self.use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning(f"No checkpoint path was specified. Continue with ImageNet pre-trained weights!")

        # add input normalization to the model
        self.encoder = nn.Sequential(ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), self.encoder)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        if not osp.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found!")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint["model"]
        for name, param in model_state_dict.items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[1][0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1][1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.weight_norm_dim >= 0

# ============== DomainNet 兼容的 AdaMoCo ==============
class AdaMoCoForDomainNet(nn.Module):
    def __init__(self, query_model, momentum_model, 
                 features_length, num_classes, dataset_length, temporal_length=5):
        super(AdaMoCoForDomainNet, self).__init__()

        self.m = 0.999  # momentum coefficient
        self.first_update = True

        # Query model (trainable)
        self.query_model = query_model
        
        # Momentum model (momentum updated)  
        self.momentum_model = momentum_model
        
        # Freeze momentum model
        self.momentum_model.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0
        self.T_moco = 0.07  # temperature for contrastive learning

        # Queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        # Memory buffers
        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer("labels", torch.randint(0, num_classes, (self.K,)))
        self.register_buffer("idxs", torch.randint(0, dataset_length, (self.K,)))
        self.register_buffer("mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length)))
        self.register_buffer("real_labels", torch.randint(0, num_classes, (dataset_length,)))

        self.features = F.normalize(self.features, dim=0)
        
        # Move to GPU
        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.query_model.parameters(), self.momentum_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        """Update memory queue and temporal labels"""
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label
        self.queue_ptr = end % self.K

        # Update temporal memory
        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        """Get memory features and labels"""
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Forward pass for AdaMoCo with DomainNet model
        Args:
            im_q: query images (weak augmentation)
            im_k: key images (strong augmentation), optional
            cls_only: if True, only return classification features and logits
        """
        # Query path
        feats_q, logits_q = self.query_model(im_q, return_feats=True)

        if cls_only:
            return feats_q, logits_q

        # Normalize query features for contrastive learning
        q = F.normalize(feats_q, dim=1)

        # Key path (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update momentum model
            
            k_feats, _ = self.momentum_model(im_k, return_feats=True)  # key features
            k = F.normalize(k_feats, dim=1)  # normalized key features

        # Compute contrastive logits
        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # Negative logits: NxK  
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])

        # Concatenate positive and negative logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits_ins /= self.T_moco

        return feats_q, logits_q, logits_ins, k

# 筛选假负例
@torch.no_grad()
def identify_false_negatives(anchor_feats, support_feats, memory_feats, top_k=5, threshold=0.7, agg='max', block_size=2048):
    """
    Identify candidate false negatives for each anchor using support views (vectorized, memory-friendly).

    Args:
        anchor_feats: [N, D]
        support_feats: [N, S, D]
        memory_feats: [K, D]
        top_k: maximum number of candidates
        threshold: similarity threshold
        agg: 'max' or 'mean' for support view aggregation
        block_size: split memory into blocks to save GPU memory
    Returns:
        false_negatives: list of tensors (indices in memory queue)
    """
    N, S, D = support_feats.shape
    K = memory_feats.shape[0]

    # 归一化
    support = F.normalize(support_feats, dim=2)  # [N, S, D]
    memory = F.normalize(memory_feats, dim=1)    # [K, D]

    sim = torch.zeros(N, K, device=anchor_feats.device)

    for start in range(0, K, block_size):
        end = min(start + block_size, K)
        mem_block = memory[start:end]  # [B, D]
        # [N, S, B]
        sim_block = torch.einsum('nsd,bd->nsb', support, mem_block)

        # 聚合 support views
        if agg == 'max':
            sim[:, start:end] = sim_block.max(1)[0]  # max over support views
        else:
            sim[:, start:end] = sim_block.mean(1)    # mean over support views

    # 根据 threshold 和 top_k 筛选假负例
    false_negatives = []
    for i in range(N):
        score = sim[i] # [-1,1]
        candidates = (score >= threshold).nonzero(as_tuple=False).squeeze(1)
        if top_k > 0 and candidates.numel() > top_k:
            topk_vals, topk_idx = score[candidates].topk(top_k)
            candidates = candidates[topk_idx]
        false_negatives.append(candidates)

    return false_negatives

def contrastive_loss(logits_ins, pseudo_labels, mem_labels, false_negatives=None, strategy='elimination'):
    """
    logits_ins: [N, 1+K]  第一列是正样本
    false_negatives: list[Tensor]  每个 tensor 是 memory 里 0-based 索引
    strategy: 'elimination' | 'attraction'
    """
    logits = logits_ins
    N, K_p1 = logits.shape
    device = logits.device

    # --------- 构造正样本 mask ---------
    pos_mask = torch.zeros_like(logits, dtype=torch.bool)
    pos_mask[:, 0] = True                       # 原始正样本
    if false_negatives is not None and strategy == 'attraction':
        for i, fn in enumerate(false_negatives):
            if fn.numel() > 0:
                pos_mask[i, 1 + fn.clamp(max=K_p1-1)] = True   # 假负例也当正

    # --------- elimination：把假负例 logit 置 -inf ---------
    if strategy == 'elimination' and false_negatives is not None:
        elim_mask = torch.zeros_like(logits, dtype=torch.bool)
        for i, fn in enumerate(false_negatives):
            if fn.numel() > 0:
                elim_mask[i, 1 + fn.clamp(max=K_p1-1)] = True
        logits = logits.masked_fill(elim_mask, torch.finfo(logits.dtype).min)  # -inf
      #  print("相似度排除策略")

        

    # --------- 多正对比损失（attraction） ---------
    if strategy == 'attraction':
        # 分子：所有正样本 exp 之和
        # pos_exp = (logits * pos_mask).exp().sum(1)
        # 分母：全部 exp 之和
        # all_exp = logits.exp().sum(1)
        # loss = -torch.log(pos_exp / all_exp).mean()
        loss = -torch.log((logits * pos_mask).exp().sum(1) / logits.exp().sum(1)).mean()
    else:
      #  print("历史排除策略")
        # 历史伪标签去除负样本
        # labels_ins = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # Create mask to exclude false negatives
        mask = torch.ones_like(logits, dtype=torch.bool)
        # Exclude samples that have same label in any historical epoch
        mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2)
        # Apply mask (set excluded logits to -inf)
        logits = torch.where(mask, logits, torch.tensor(float("-inf")).cuda())
        
        # loss = F.cross_entropy(logits, labels_ins)

        # 普通交叉熵（elimination 已 mask 掉假负例）
        labels = torch.zeros(N, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)

    return loss

def nl_criterion(output, y, num_classes):
    """
    Negative Learning Loss: train model to not predict random negative labels
    """
    output = torch.log(torch.clamp(1. - F.softmax(output, dim=1), min=1e-5, max=1.))
    
    # Generate random complementary labels
    labels_neg = ((y.unsqueeze(-1) + 
                   torch.LongTensor(len(y), 1).random_(1, num_classes).cuda()) 
                  % num_classes).view(-1)
    
    l = F.nll_loss(output, labels_neg, reduction='none')
    return l

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# ============== 数据增强策略 ==============
def get_augmentation(aug_type, normalize=True):
    if normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if aug_type == "moco-v2":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == "plain":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == "test":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return None

def get_augmentation_domainnet126(aug_type):
    if aug_type == "moco-v2":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif aug_type == "plain":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    elif aug_type == "test":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    return None

def get_augmentation_versions(cfg):
    """获取多种增强版本：t(测试), w(弱增强), s(强增强1), s(强增强2)"""
    transform_list = []
    # 从 cfg 读取 N 参数
    N = getattr(cfg.UAHLR, "N", 4)
    # 动态生成版本字符串
    versions = "twss" + "s" * N
    print("versions:",versions)
    if cfg.SETTING.DATASET=="domainnet126":
        for version in versions:
            if version == "s":
                transform_list.append(get_augmentation_domainnet126("moco-v2"))
            elif version == "w":
                transform_list.append(get_augmentation_domainnet126("plain"))
            elif version == 't':
                transform_list.append(get_augmentation_domainnet126("test"))
            else:
                raise NotImplementedError(f"{version} version not implemented.")
    else:
        for version in versions:
            if version == "s":
                transform_list.append(get_augmentation("moco-v2"))
            elif version == "w":
                transform_list.append(get_augmentation("plain"))
            elif version == 't':
                transform_list.append(get_augmentation("test"))
            else:
                raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)
    return transform

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
    #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def image_test_domainnet126(resize_size=256, crop_size=224, alexnet=False):
#   if not alexnet:
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
  #else:
    #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # normalize
    ])

def data_load(cfg): 
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    
    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    # 使用新的增强策略
    train_transform = get_augmentation_versions(cfg)
    
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=train_transform)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    if cfg.SETTING.DATASET=="domainnet126":
        dsets["test"] = ImageList_idx_aug_fix(txt_test, transform=image_test_domainnet126())
    else:
        dsets["test"] = ImageList_idx_aug_fix(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, moco_model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]  
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = moco_model(inputs, cls_only=True)  # Use AdaMoCo
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

# ============== 保持原有的伪标签细化等函数不变 ==============
def entropy(p, axis=1):
    """计算概率分布的熵（不确定性）"""
    return -torch.sum(p * torch.log2(p + 1e-5), dim=axis)

def get_distances(X, Y, dist_type="cosine"):
    """计算特征距离矩阵"""
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1),
                                     F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")
    return distances



@torch.no_grad()
def soft_k_nearest_neighbors(
    features,
    features_bank,
    probs_bank,
    num_neighbors=10,
    eps=1e-8
):
    """
    基于熵不确定性加权的 Soft KNN 伪标签细化

    Args:
        features:        (N, D) 当前目标样本特征
        features_bank:   (M, D) 目标域特征库
        probs_bank:      (M, C) 特征库中样本的预测概率分布
        num_neighbors:   K
        eps:             防止 log(0)

    Returns:
        pred_labels:     (N,)   细化后的伪标签
        pred_probs:      (N, C) 加权聚合后的类别概率
    """

    pred_probs = []

    # 归一化特征，用于余弦相似度
    features = F.normalize(features, dim=1)
    features_bank = F.normalize(features_bank, dim=1)

    # 分批处理，防止显存溢出
    for feats in features.split(64):
        # (B, M) 余弦距离 = 1 - cosine similarity
        sim = torch.mm(feats, features_bank.t())
        distances = 1.0 - sim

        # 选取 K 个最近邻
        _, idxs = distances.sort(dim=1)
        idxs = idxs[:, :num_neighbors]  # (B, K)

        # (B, K, C) 邻居的预测概率
        neighbor_probs = probs_bank[idxs]

        # -------- 不确定性建模（熵） --------
        # H(p_i) = - sum_k p_i(k) log p_i(k)
        entropy = -torch.sum(
            neighbor_probs * torch.log(neighbor_probs + eps),
            dim=-1
        )  # (B, K)

        # η_i = exp(-H(p_i))
        weights = torch.exp(-entropy)  # (B, K)

        # 归一化权重
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # (B, K)

        # -------- 加权软投票 --------
        # \hat{p}_t = sum_i η_i p_i
        probs = torch.sum(
            neighbor_probs * weights.unsqueeze(-1),
            dim=1
        )  # (B, C)

        pred_probs.append(probs)

    # (N, C)
    pred_probs = torch.cat(pred_probs, dim=0)

    # 伪标签
    pred_labels = pred_probs.argmax(dim=1)

    return pred_labels, pred_probs



def div_regularization(logits, epsilon=1e-8):
    """防止塌缩的正则化项"""
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))
    return loss_div

# ============== 修改后的训练函数支持DomainNet ==============
def train_target(cfg):
    # 检测是否是DomainNet数据集
    is_domainnet = getattr(cfg, 'IS_DOMAINNET', False) or 'domainnet' in cfg.SETTING.DATASET.lower()
    
    # 获取当前时间（格式：年-月-日_时-分-秒）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 拼接保存路径
    output_target_dir = cfg.output_dir
    tensorboard_dir_name = "tensorboard"
    pth_name = "pth"
    if not osp.exists(osp.join(output_target_dir, pth_name)):
        os.makedirs(osp.join(output_target_dir, pth_name))
    writer = SummaryWriter(log_dir= osp.join(output_target_dir, tensorboard_dir_name))

    clip_model, preprocess,_ = clip.load(cfg.UAHLR.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)  # 文本 prompt: "a photo of a class"
    dset_loaders = data_load(cfg)
    
    if is_domainnet:
        # DomainNet: 使用统一的网络结构
        logger.info("Using DomainNet architecture")
        
        pth = osp.join(cfg.output_dir_src,'best_' + cfg.domain[cfg.SETTING.S] +'_2020.pth')
        # 查询网络（可训练）
        query_model = ResNetDomainNet126(
            arch=cfg.MODEL.ARCH, 
            checkpoint_path=pth,
            num_classes=cfg.class_num,
            bottleneck_dim=cfg.bottleneck
        ).cuda()
        
        # 动量网络（用于对比学习）
        momentum_model = ResNetDomainNet126(
            arch=cfg.MODEL.ARCH, 
            checkpoint_path=pth,
            num_classes=cfg.class_num,
            bottleneck_dim=cfg.bottleneck
        ).cuda()
        
        # 创建AdaMoCo模型
        num_sample = len(dset_loaders["target"].dataset)
        moco_model = AdaMoCoForDomainNet(
            query_model=query_model,
            momentum_model=momentum_model,
            features_length=query_model.output_dim,
            num_classes=cfg.class_num,
            dataset_length=num_sample,
            temporal_length=1
        ).cuda()
        
        # 冻结分类器用于适配
        query_model.fc.eval()
        momentum_model.fc.eval()
        for param in query_model.fc.parameters():
            param.requires_grad = False
        for param in momentum_model.fc.parameters():
            param.requires_grad = False
            

        param_group = []
        for k, v in query_model.named_parameters():
            if 'netC' in k or 'fc' in k:
                v.requires_grad = False
            else:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]
        
    else:
        # 原有架构：分层结构 netF, netB, netC
        logger.info("Using standard architecture (netF + netB + netC)")
        
        if cfg.MODEL.ARCH[0:3] == 'res':
            netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda() 
            momentum_netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
        elif cfg.MODEL.ARCH[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()
            momentum_netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

        netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
        netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
        
        momentum_netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
        momentum_netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
        
        # 载入源域预训练权重
        modelpath = cfg.output_dir_src + '/source_F.pt'   
        netF.load_state_dict(torch.load(modelpath))
        momentum_netF.load_state_dict(torch.load(modelpath))
        
        modelpath = cfg.output_dir_src + '/source_B.pt'   
        netB.load_state_dict(torch.load(modelpath))
        momentum_netB.load_state_dict(torch.load(modelpath))
        
        modelpath = cfg.output_dir_src + '/source_C.pt'    
        netC.load_state_dict(torch.load(modelpath))
        momentum_netC.load_state_dict(torch.load(modelpath))
        
        # 创建原始AdaMoCo模型
        num_sample = len(dset_loaders["target"].dataset)
        moco_model = AdaMoCo(
            netF=netF, netB=netB, netC=netC,
            momentum_netF=momentum_netF, momentum_netB=momentum_netB, momentum_netC=momentum_netC,
            features_length=cfg.bottleneck,
            num_classes=cfg.class_num,
            dataset_length=num_sample,
            temporal_length=1
        ).cuda()

        # Freeze classifier for adaptation
        netC.eval()
        momentum_netC.eval()
        for k, v in netC.named_parameters():
            v.requires_grad = False
        for k, v in momentum_netC.named_parameters():
            v.requires_grad = False

        # 设置优化器参数组
        param_group = []
        for k, v in netF.named_parameters():
            if cfg.OPTIM.LR_DECAY1 > 0:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
            else:
                v.requires_grad = False
        for k, v in netB.named_parameters():
            if cfg.OPTIM.LR_DECAY2 > 0:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
            else:
                v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    # ---------- 初始化记忆库 score_bank ----------
    loader = dset_loaders["target"]
    score_bank = torch.randn(num_sample, cfg.class_num).cuda() 
    if is_domainnet:
        feature_bank = torch.randn(num_sample, query_model.output_dim).cuda()
    else:
        feature_bank = torch.randn(num_sample, cfg.bottleneck).cuda()

    # Initialize banks with source model predictions
    if is_domainnet:
        query_model.eval()
    else:
        netF.eval()
        netB.eval() 
        netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][1] # 使用弱增强
            indx = data[-2]
            inputs = inputs.cuda()
            features, outputs = moco_model(inputs, cls_only=True)
            outputs = nn.Softmax(dim=1)(outputs)
            score_bank[indx] = outputs.detach().clone()
            feature_bank[indx] = features.detach().clone()

    # 迭代训练
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    text_features = None
    best_acc = 0.0
    epoch = 0

    # 伪标签优化配置参数
    use_label_refinement = getattr(cfg, 'USE_LABEL_REFINEMENT', True)
    num_neighbors = getattr(cfg.UAHLR, 'NUM_NEIGHBORS', 4)
    print("num_neighbors:",num_neighbors)
    use_contrastive = getattr(cfg, 'USE_CONTRASTIVE', True)
    use_negative_learning = getattr(cfg, 'USE_NEGATIVE_LEARNING', False)
    threshold = getattr(cfg.UAHLR,'THRESHOLD',0.8)
    print("threshold:",threshold)
    CUR_RATIO = getattr(cfg.UAHLR, "CUR_RATIO", 0.3)
    print("课程学习比例：",CUR_RATIO)


    # 训练PEM
    f = f"PEM_{cfg.SETTING.DATASET}_{cfg.SETTING.S}_{cfg.SETTING.T}.pth"
    if not osp.exists("PEM_ckpt/"):
        os.makedirs("PEM_ckpt/")
    p = os.path.join("PEM_ckpt/", f)
    root, ext = os.path.splitext(p)
    best_p = root + "_best" + ext
    # train_ProbVLM(clip_model, dset_loaders['target'], dset_loaders['test'], device="cuda",init_lr=1e-4,num_epochs=50,ckpt_path=p,T1=1e0,T2=5e-2,epoch_index=None,writer=writer)

    while iter_num < max_iter:
        try:
            img, y, tar_idx,_ = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            img, y, tar_idx,_ = next(iter_test)
        inputs_test = img[1] # 弱增强
        strong_x = img[2] # 用于对比学习，拉近同样本特征，清晰分类边界
        strong_x2 = img[3]
        support_views = img[4:] # 额外增强，用来识别假负例

        if inputs_test.size(0) == 1:
            continue
            
        # 每个epoch更新一次伪标签和文本特征
        if iter_num % interval_iter == 0:
            epoch = iter_num // len(dset_loaders["target"])
            
            if cfg.UAHLR.CLS_PAR > 0:
                if is_domainnet:
                    query_model.eval()
                else:
                    netF.eval()
                    netB.eval() 
                    netC.eval()

                # 课程学习权重计算
                warmup_epoch = int(cfg.TEST.MAX_EPOCH * CUR_RATIO)
                epoch_ratio = min((epoch + 1) / warmup_epoch, 1.0)
                
                # 训练概率适配器
                if epoch+1<=warmup_epoch and (epoch+1)%10==1:
                    train_PEM(moco_model, dset_loaders['target'], dset_loaders['test'], device="cuda",init_lr=1e-3,num_epochs=10+40,ckpt_path=p,T1=0,T2=1.0,epoch_index=None,writer=writer)

                confi_imag, confi_dis, clip_all_output, all_aleatoric_weight = obtain_label_enhanced_with_adapter(
                    dset_loaders['test'], moco_model, text_inputs, text_features, clip_model,
                    feature_bank, score_bank, use_label_refinement, num_neighbors, epoch_ratio, best_p)

                clip_all_output = clip_all_output.cuda()
                text_features = prompt_tuning.prompt_main(cfg, confi_imag, confi_dis, iter_num, writer)
                cfg.load = 'prompt_model.pt'
                
                if is_domainnet:
                    query_model.train()
                    # 保持分类器冻结
                    query_model.fc.eval()
                else:
                    netF.train()
                    netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_test = inputs_test.cuda()
        strong_x = strong_x.cuda()
        strong_x2 = strong_x2.cuda()
        
        all_aleatoric_weight = all_aleatoric_weight.cuda()

        # Forward pass with weak augmentation (for pseudo-label generation)
        features_test, outputs_test = moco_model(inputs_test, cls_only=True)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        # Get refined pseudo labels
        refined_labels, refined_probs = soft_k_nearest_neighbors(features_test.cuda(), feature_bank, score_bank, num_neighbors)
        
        # Forward pass with strong augmentation (for contrastive learning)
        if use_contrastive:
            _, logits_strong, logits_ctr, keys = moco_model(strong_x, strong_x2)

            # --- 假负例识别 ---
            with torch.no_grad():
                support_feats_list = []
                # if not isinstance(support_views, (list, tuple)):
                #     support_views = [support_views]
                for img_view in support_views:   # support_views 是一个 list，长度 = num_support
                    img_view = img_view.cuda()
                    feats, _ = moco_model(img_view, cls_only=True)
                    support_feats_list.append(feats.unsqueeze(1))   # [N, 1, D]
                # 拼接成 [N, S, D]
                support_feats = torch.cat(support_feats_list, dim=1)

                # 进入假负例识别
                false_negatives = identify_false_negatives(
                    anchor_feats=features_test,                       # [N, D]
                    support_feats=support_feats,                # [N, S, D]
                    memory_feats=moco_model.features.T,         # [K, D]
                    top_k=0,
                    threshold=threshold,
                    agg='max'
                )
              #  print("------------------------")
                #print(false_negatives)
            loss_contrastive = contrastive_loss(
                logits_ins=logits_ctr,
                pseudo_labels=moco_model.mem_labels[tar_idx],
                mem_labels=moco_model.mem_labels[moco_model.idxs],
                false_negatives=false_negatives,
                # false_negatives=None,
                strategy="elimination"
            )

            # Update MoCo memory
            tar_idx_cuda = tar_idx.cuda()  # 将 tar_idx 转换到 GPU
            y_cuda = y.cuda()
            moco_model.update_memory(epoch, tar_idx_cuda, keys, refined_labels, y_cuda)  
        else:
            loss_contrastive = torch.tensor(0.0).cuda()
        
        CTR_PAR = getattr(cfg.UAHLR, "CTR_PAR", 1.0)
        # print("CTR_PAR",CTR_PAR)
        loss_contrastive *= CTR_PAR

        # Generate mixed pseudo labels
        ln_sam = softmax_out.shape[0]
        K = softmax_out.size(1)
        _, clip_predict = torch.max(clip_all_output[tar_idx], 1)
        clip_one = np.eye(K)[clip_predict.cpu()]
        refined_one = np.eye(K)[refined_labels.cpu().numpy()]
        
        # Confidence-based fusion
        refined_conf = refined_probs.max(dim=1, keepdim=True)[0]
        clip_conf = clip_all_output[tar_idx].max(dim=1, keepdim=True)[0]
        w = (refined_conf / (refined_conf + clip_conf + 1e-6))
        w = w.cpu().numpy()
        refined_mix = w * refined_one + (1 - w) * clip_one
        refined_mix = torch.from_numpy(refined_mix).cuda()

        # ---------- 损失计算 ----------
        if cfg.UAHLR.CLS_PAR > 0:
            targets = refined_mix.detach()
            
            if use_negative_learning:
                # Use negative learning loss
                print("Use negative learning loss")
                loss_soft = nl_criterion(outputs_test, refined_labels, cfg.class_num)
                loss_soft = loss_soft * all_aleatoric_weight[tar_idx]
                classifier_loss = loss_soft.mean()
            else:
                # Standard cross entropy loss
                loss_soft = (- targets * outputs_test).sum(dim=1)
                loss_soft = loss_soft * all_aleatoric_weight[tar_idx]
                classifier_loss = loss_soft.mean()
            
            classifier_loss *= cfg.UAHLR.CLS_PAR
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        # 互信息一致性损失
        iic_loss = cfg.UAHLR.IIC_PAR * IID_losses.IID_loss(softmax_out, clip_all_output[tar_idx]) 

        # 熵正则（防止塌缩）
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = - cfg.UAHLR.GENT_PAR * (torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON)))

        total_loss = classifier_loss + iic_loss + gentropy_loss + loss_contrastive

        writer.add_scalar('loss/total', total_loss.detach().item(), iter_num)
        writer.add_scalar('loss/cls', classifier_loss.detach().item(), iter_num)
        writer.add_scalar('loss/iic2', iic_loss.detach().item(), iter_num)
        writer.add_scalar('loss/gentropy', gentropy_loss.detach().item(), iter_num)
        writer.add_scalar('loss/contrastive', loss_contrastive.detach().item() if isinstance(loss_contrastive, torch.Tensor) else 0, iter_num)

        # Update memory banks
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()
            feature_bank[tar_idx] = features_test.detach().clone()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            if is_domainnet:
                query_model.eval()
            else:
                netF.eval()
                netB.eval()
                
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], moco_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,total_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], moco_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,total_loss)
            logger.info(log_str)
            
            if acc_s_te > best_acc:
                best_acc = acc_s_te
                # Save the best networks
                if is_domainnet:
                    torch.save(query_model.state_dict(), osp.join(output_target_dir, pth_name, "best_target_model_" + cfg.savename + ".pt"))
                else:
                    torch.save(netF.state_dict(), osp.join(output_target_dir, pth_name, "best_target_F_" + cfg.savename + ".pt"))
                    torch.save(netB.state_dict(), osp.join(output_target_dir, pth_name, "best_target_B_" + cfg.savename + ".pt"))
                    torch.save(netC.state_dict(), osp.join(output_target_dir, pth_name, "best_target_C_" + cfg.savename + ".pt"))
                logger.info("Best accuracy: {:.2f}%".format(best_acc))
            writer.add_scalar('acc/test_acc', acc_s_te, iter_num)
            writer.add_scalar('acc/best_acc', best_acc, iter_num)

            if is_domainnet:
                query_model.train()
                query_model.fc.eval()  # 保持分类器冻结
            else:
                netF.train()
                netB.train()
    
    logger.info("Best accuracy: {:.2f}%".format(best_acc))
    # ====== 关键：重命名日志文件 ======
    try:
        old_log = osp.join(cfg.output_dir, "log.txt")
        new_log = osp.join(cfg.output_dir, f"log_best_{best_acc:.2f}.txt")
        os.rename(old_log, new_log)
        logger.info(f"日志文件已重命名为: {new_log}")
    except FileNotFoundError:
        logger.warning("log.txt 不存在，无法重命名。")
    writer.close()

    if cfg.ISSAVE:   
        if is_domainnet:
            torch.save(query_model.state_dict(), osp.join(output_target_dir, pth_name, "target_model_" + cfg.savename + ".pt"))
        else:
            torch.save(netF.state_dict(), osp.join(output_target_dir, pth_name, "target_F_" + cfg.savename + ".pt"))
            torch.save(netB.state_dict(), osp.join(output_target_dir, pth_name, "target_B_" + cfg.savename + ".pt"))
            torch.save(netC.state_dict(), osp.join(output_target_dir, pth_name, "target_C_" + cfg.savename + ".pt"))
    

    return best_acc

def obtain_label_enhanced_with_adapter(loader, moco_model, text_inputs, text_features, clip_model,
                         feature_bank, score_bank, use_label_refinement=True, num_neighbors=10, epoch_ratio=1.0, path=None):
    """增强版标签获取，集成伪标签细化，使用AdaMoCo模型"""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0] # 要用测试增强视图
            labels = data[1]
            inputs_clip = data[-1]
            inputs = inputs.cuda()
            inputs_clip = inputs_clip.cuda() 
            feas, outputs = moco_model(inputs, cls_only=True)  # Use AdaMoCo
            
            if (text_features!=None):
                clip_score = clip_text(clip_model,text_features,inputs_clip)
            else :
                clip_score,_ = clip_model(inputs_clip, text_inputs)

            # ============== 新增：概率适配器特征分布 ==============
            mu, img_alpha, img_beta = extract_PEM_features(moco_model,inputs,ckpt_path=path)
            GGuncer = get_GGuncer(img_alpha, img_beta)
            aleatoric_uncertainty = GGuncer.mean(dim=1)
            u = aleatoric_uncertainty
            u_log = torch.log1p(u)                   # log(1+x) 防止 0
            u_norm = (u_log - u_log.min()) / (u_log.max() - u_log.min() + 1e-8)
            aleatoric_weight = torch.exp(-u_norm)
            # 课程学习平滑
            aleatoric_weight = aleatoric_weight * (1 - epoch_ratio) + 1.0 * epoch_ratio

            clip_score = clip_score.cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float().cpu()
                all_features = feas.float().cpu()
                all_aleatoric_weight = aleatoric_weight.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                all_features = torch.cat((all_features, feas.float().cpu()), 0)
                all_aleatoric_weight = torch.cat((all_aleatoric_weight, aleatoric_weight.cpu()), 0)
                
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    _, predict_clip = torch.max(clip_all_output, 1)  
    accuracy_clip = torch.sum(torch.squeeze(predict_clip).float() == all_label).item() / float(all_label.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    
    # ============== 伪标签细化 ==============
    if use_label_refinement:
        refined_labels, refined_probs = soft_k_nearest_neighbors(
            all_features.cuda(), feature_bank, score_bank, num_neighbors)
        
        alpha = 1
        all_mix_output = alpha * refined_probs.cpu() + (1-alpha) * all_output
        all_mix_output = (all_mix_output + clip_all_output) / 2
    else:
        all_mix_output = (all_output + clip_all_output) / 2




    confi_dis = all_mix_output.detach()
    confi_imag = loader.dataset.imgs
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    log_str = 'Accuracy = {:.2f}% -> CLIP_Accuracy  = {:.2f}%'.format(accuracy * 100, accuracy_clip * 100)
    if use_label_refinement:
        refined_acc = torch.sum(refined_labels.cpu() == all_label).item() / float(all_label.size()[0])
        log_str += ' -> Refined_Accuracy = {:.2f}%'.format(refined_acc * 100)
    
    logging.info(log_str)
    return confi_imag, confi_dis, clip_all_output, all_aleatoric_weight

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.UAHLR.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

def clip_text(model,text_features,inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits

