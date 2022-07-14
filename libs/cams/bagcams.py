from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from libs.cams.basic import _BaseWrapper

class BagCAMs(_BaseWrapper):

    def __init__(self, extractor, classifier):
        self.fmap_pool = {}
        self.fmap_pool_in = {}
        self.grad_pool = {}
        self.grad_pool_in = {}
    
        super(BagCAMs, self).__init__(extractor, classifier)
        
        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

                if isinstance(module, nn.ReLU):
                    return (F.relu(grad_in[0]),)
                        
            return backward_hook
        
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()
                self.fmap_pool_in[key] = input[0].detach()

            return forward_hook

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_forward_hook(save_fmaps(module[0])))
            self.handlers.append(module[1].register_backward_hook(save_grads(module[0])))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        self.ids = ids
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        ##
        self.phi = torch.zeros(self.logits.shape[0], 1).cuda()
        for i in range(0, self.logits.shape[0]):
            self.phi[i] = self.logits[i, ids[i]]
        # self.logits[:, ids]
        ##
        #print(one_hot)
        self.logits.log().backward(gradient=one_hot, retain_graph=True)

    def generate(self, target_layer):

        ##obtain the gradient
        grads = self._find(self.grad_pool, target_layer)

        ##obtain the feature map
        features = self._find(self.fmap_pool, target_layer)

        ##Calculate BagCAMs
        term_2 = grads*features
        term_1 = grads*features + 1
        term_1 = F.adaptive_avg_pool2d(term_1, 1) #sum_m
        bagcams = torch.relu(torch.mul(term_1, term_2)).sum(dim=1, keepdim=True) #sum_c

        ##Upsampling to Original Size of Images
        bagcams = F.interpolate(
            bagcams, self.image_shape, mode="bilinear", align_corners=False
        )
        
        ##Normalized the localization Maps
        B, C, H, W = bagcams.shape
        bagcams = bagcams.view(B, -1)
        bagcams -= bagcams.min(dim=1, keepdim=True)[0]
        bagcams /= bagcams.max(dim=1, keepdim=True)[0]
        bagcams = bagcams.view(B, C, H, W)

        return bagcams