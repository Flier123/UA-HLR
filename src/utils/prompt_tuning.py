from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from src.utils import IID_losses
from clip.custom_clip import get_coop
from data.datautils_domain import  build_dataset
from data.cls_to_names import *
from data.domain_datasets import domain_datasets


def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=[0.26862954, 0.26130258, 0.27577711])
                            
    return transforms.Compose([
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

def test_time_tuning(model, inputs,pesu_label, optimizer, cfg,iter_num,writer=None):
    for j in range(cfg.DIFO.TTA_STEPS):
        with torch.cuda.amp.autocast():
            output,_ = model(inputs) 
            pesu_label = pesu_label.cuda()
            output = nn.Softmax(dim=1)(output)
            loss = IID_losses.IID_loss(output, pesu_label) # 互信息最大化（Invariant Information Distillation, IID），用来对齐模型预测和伪标签分布
        # if writer is not None:
        #     writer.add_scalar('loss/iic1', loss.item(), iter_num)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 

def prompt_main(cfg,confi_imag,confi_dis,iter_num,writer=None):
    # This codebase has only been tested under the single GPU setting
    assert int(cfg.GPU_ID) is not None
    text_features = main_worker(cfg,confi_imag,confi_dis,iter_num,writer)
    text_features = text_features.detach()
    return text_features


def main_worker(cfg,confi_imag,confi_dis,iter_num,writer=None):
    # 如果当前数据集在 domain_datasets 中，则读取对应 domain 信息
    if cfg.SETTING.DATASET in domain_datasets:
        cfg.domain_name = cfg.domain[cfg.SETTING.T]
        classnames = cfg.classname

    # 初始化 CoOp 模型（CLIP + 可学习的 prompt）
    model = get_coop(cfg.DIFO.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.DIFO.N_CTX, cfg.DIFO.CTX_INIT)
    model = model.cuda()

    # 如果配置中指定了加载的 prompt 参数则加载prompt, 默认为None
    if cfg.DIFO.LOAD is not None:
        print("loading prompt")
        pretrained_ctx = torch.load(cfg.DIFO.LOAD)['ctx']
        assert pretrained_ctx.size()[0] == cfg.DIFO.N_CTX
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx
    # 冻结除 prompt 以外的所有参数，只更新 prompt
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_param = model.prompt_learner.parameters() # 获取 prompt 的可训练参数
    if 'RN' in cfg.DIFO.ARCH :
        prompt_lr = cfg.OPTIM.LR*0.1 # ResNet 用更小学习率
        data_transform = image_test_50()
    else :
        data_transform = image_test()
        prompt_lr = cfg.OPTIM.LR

    optimizer = torch.optim.SGD(trainable_param, prompt_lr,weight_decay=5e-4,momentum=0.9,nesterov=False)
    optim_state = deepcopy(optimizer.state_dict())
    cudnn.benchmark = True
    set_id = 'sfuda'
    model.reset_classnames(classnames, cfg.DIFO.ARCH)
    # val_loader (图像, 真实标签, 伪标签（来自confi_dis）, 索引)
    val_dataset = build_dataset(set_id, data_transform,confi_imag,confi_dis,cfg.DATA_DIR,cfg.domain_name,mode='test')
    batchsize = cfg.TEST.BATCH_SIZE
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batchsize, shuffle=True,
                num_workers=cfg.NUM_WORKERS,drop_last=False) 
    # 在验证集（这里验证集没有和target划分，是同一份）上进行 Test-Time Adaptation，并返回最终的 text_features   
    text_features = test_time_adapt_eval(val_loader, model, optimizer, optim_state, cfg, iter_num, writer)
    return text_features

def test_time_adapt_eval(val_loader, model, optimizer, optim_state, cfg,iter_num2, writer=None):
    """
        测试时自适应
        遍历验证集 batch；
        每个 batch 用 pesu_label 做一次 Test-Time Adaptation（比如更新 prompt / BN）；
        适应后推理，提取 text_features；
        最终保存 prompt 模型参数，并返回最后一批的 text_features。
    """
    with torch.no_grad():
        model.train()
    max_iter = len(val_loader) # 总迭代次数
    iter_num = 0
    while iter_num < max_iter:
        try:
            images, target,pesu_label,_ = next(iter_test) # 从 dataloader 中取数据
        except:
            iter_test = iter(val_loader)
            images, target,pesu_label,_ = next(iter_test)

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda(int(cfg.GPU_ID), non_blocking=True)
        image = images
        target = target.cuda(int(cfg.GPU_ID), non_blocking=True)
        
        if cfg.DIFO.TTA_STEPS > 0: # 如果配置了 TTA steps，说明要做 prompt 更新。默认为1
            with torch.no_grad():
                model.train()
        optimizer.load_state_dict(optim_state) # 每次都重新学习？
        # 用伪标签做一次 test-time tuning（如最大化互信息，对齐CoOp预测分布与伪标签分布）
        test_time_tuning(model,images,pesu_label, optimizer, cfg,iter_num2,writer)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model.eval()
                _,text_features = model(image)
        iter_num = iter_num + 1
    torch.save(model.prompt_learner.state_dict(),"prompt_model.pt")  # 保存最终学到的 prompt 参数
    return text_features