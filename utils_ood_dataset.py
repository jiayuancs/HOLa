"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
from hico_text_label import hico_unseen_index, HOI_TO_AO, ACT_IDX_TO_ACT_NAME, obj_to_name, OBJ_IDX_TO_COCO_ID, MAP_AO_TO_HOI
from vcoco_text_label import MAP_AO_TO_HOI_COCO
import sys
sys.path.append('../pocket/pocket')
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
sys.path.append('detr')
import detr.datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from util import box_ops
from PIL import Image
from hicodet.static_hico import HICO_INTERACTIONS
from hico_text_label import HICO_INTERACTIONS, hico_unseen_index 
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

def custom_collate(batch):
    images = []
    targets = []
    # images_clip = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
        
        # images_clip.append(im_clip)
    return images, targets

class DataFactoryOOD(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name, zero_shot=False, zs_type='rare_first', num_classes=600, detr_backbone="R50", syn = None): ##ViT-B/16, ViT-L/14@336px
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16',  'ViT-B/32']
        self.clip_model_name = clip_model_name
        if self.clip_model_name == 'ViT-B/16' or self.clip_model_name == 'ViT-B/32':
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336

        self.dataset = HICODet(
            root="/workspace/dataset/swig-hoi/images_512",
            anno_file="/workspace/dataset/swig-hoi/tmp/swig_hico.json",
            object_cls_num=80,
            verb_cls_num=305,
            hoi_cls_num=1161,
            target_transform=pocket.ops.ToTensor(input_format='dict')
        )

        # add clip normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
 
        self.transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
        self.clip_transforms = T.Compose([
            T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
        ])
        
        self.partition = partition
        self.name = name
        self.count=0
        self.zero_shot = zero_shot
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]

        device = "cuda"
        # _, self.process = clip.load('ViT-B/16', device=device)
        print(self.clip_model_name)
        # _, self.process = clip.load(self.clip_model_name, device=device)

        self.keep = [i for i in range(len(self.dataset))]

        
    def __len__(self):
        return len(self.keep)
        return len(self.dataset)

    # train detr with roi
    def __getitem__(self, i):
        (image, target), filename = self.dataset[self.keep[i]]
        # (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])

        target['labels'] = target['verb']
        # Convert ground truth boxes to zero-based index and the
        # representation from pixel indices to coordinates
        # target['boxes_h'][:, :2] -= 1
        # target['boxes_o'][:, :2] -= 1

        
        image, target = self.transforms(image, target)
        image_clip, target = self.clip_transforms(image, target)  
        image, _ = self.normalize(image, None)
        image_clip, target = self.normalize(image_clip, target)
        target['filename'] = filename

        return (image,image_clip), target
        image_0, target_0 = self.transforms[0](image, target)
        image, _ = self.transforms[1](image_0, None)

        target_0['valid_size'] = torch.as_tensor(image.shape[-2:])
        image_clip, target = self.transforms[3](image_0, target_0) # resize 
        image_clip, target = self.transforms[2](image_clip, target) # normlize
        if image_0.size[-1] >1344 or image_0.size[-2] >1344:print(image_0.size)
        target['filename'] = filename

        return (image,image_clip), target

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        region_props = []
        # for res in results:
        # pdb.set_trace()
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels']
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            # keep_h = keep[keep_h]
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            # keep_o = keep[keep_o]
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)
            # boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
            # continue

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # pdb.set_trace()
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)
    
        # region_props.append(dict(
        #     boxes=bx[keep],
        #     scores=sc[keep],
        #     labels=lb[keep],
        #     hidden_states=hs[keep],
        #     mask = ms[keep]
        # ))

        # return sub_boxes.int(), obj_boxes.int(), union_boxes.int()
        return sub_boxes, obj_boxes, union_boxes

    def get_union_mask(self, bbox, image_size):
        n = len(bbox)
        masks = torch.zeros
        pass


if __name__ == "__main__":
    dataset = DataFactoryOOD(
        name="hicodet",
        partition="",
        data_root="",
        clip_model_name="ViT-B/16"
    )
    print(len(dataset))
