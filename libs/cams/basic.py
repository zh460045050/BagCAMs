from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import math

class _BaseWrapper(object):
    def __init__(self, extractor, classifier):
        super(_BaseWrapper, self).__init__()
        self.device = next(extractor.parameters()).device
        self.model = extractor
        self.classifier = classifier
        self.handlers = []  # a set of hook function handlers
        self.phi = 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def _encode_one_hot(self, targets):
        #print(self.logits.shape)
        one_hot = torch.zeros_like(self.logits).to(self.device)
        for i in range(0, one_hot.shape[0]):
            one_hot[i, targets[i]] = 1.0
        return one_hot
        
    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def forward(self, image):
        self.image_shape = image.shape[2:]
        pixel_features = self.model(image)
        image_features = self.pool(pixel_features)
        self.logits = self.classifier(image_features).view(image_features.shape[0], -1)
        self.probs = F.softmax(self.logits, dim=1)
        return self.logits, self.probs  # ordered results

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
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()
