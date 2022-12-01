import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
import numpy as np

from .unet_parts import *

paths = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../')
if not os.path.exists(paths): 
    raise RuntimeError('Invalid path for descriptor tools: ' + paths)
else:
    print("ok path:", paths)
sys.path.insert(0, paths)

from models.cnn_wrapper.aslfeat import ASLFeatNet

class Our(nn.Module):
    def __init__(self, enc_channels = [1, 32, 64, 128]):
      super().__init__()
      dec_channels = enc_channels[::-1]
      self.encoder = Encoder(enc_channels)
      self.decoder = Decoder(enc_channels, dec_channels)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
      
        return out
    
    def initASLFeat(self):
        self.aslfeat = ASLFeatNet(None, False)
        self.aslfeat.init()
    
    def asl_feature_map(self, data):
        self.aslfeat.netConfig
        self.aslfeat.sess

        h, w, _ = data.shape
        
        feed_dict = {"input:0": np.expand_dims(data, 0)}
        returns = self.aslfeat.sess.run(self.aslfeat.endpoints, feed_dict=feed_dict)
        feature_maps = returns['feature_maps']
        descs = returns['descs'][0]
        score_map = returns['score_map']

        idx = 0
        d = np.zeros((h, w, 128), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                d[y][x] = descs[idx]
                idx+=1
        
        # retornando o ultimo feature map e os descritores no shape da imagem de entrada
        feature_map = feature_maps[2]
        return feature_map, d, score_map

    def init_detector(self, model_path):
        self.model_path = model_path
        self.net = Our()
        self.net.load_state_dict(torch.load(model_path, map_location = self.device))
        self.initASLFeat()


    def detect(self, img, max_kps, id=1, thresould_score=0.1): # numpy image
        if self.aslfeat == None or self.net == None:
            print("Call init_detector function before cal detect")
            return
        img2 = np.copy(img)
        img2 = img2[..., np.newaxis]
        _, descriptors, score_map_asl = self.asl_feature_map(img2)
                
        h, w = img.shape

        imT = torch.Tensor(img).unsqueeze(0).unsqueeze(0)

        h, w, _ = descriptors.shape

        score_map = self.net(imT.to(self.device))

        # nms
        local_max = F.max_pool2d(score_map, 5, stride=1, padding=2)
        is_local_max = (score_map == local_max)
        del local_max
        is_not_edge = self.edgeFilter(score_map)

        detected = torch.min(is_local_max, is_not_edge)
        detected = torch.squeeze(detected, 0)
        detected = torch.squeeze(detected, 0)

        score_map = torch.squeeze(score_map, 0)
        score_map = torch.squeeze(score_map, 0)

        score_map = score_map.cpu().detach().numpy()
        detected = detected.cpu().detach().numpy()
                
        points = score_map * detected

        filter = points > thresould_score
        
        selection = np.zeros_like(points)
        selection[filter] = points[filter]

        keypoints = []
        descs = []
        for y in range(h):
            for x in range(w):
                if selection[y][x] != 0:
                    keypoints.append([x, y, selection[y][x]])
                    descs.append(descriptors[y][x])
        
        idxs = sorted(range(len(keypoints)), key=lambda k: keypoints[k][2], reverse=True)
        
        keypoints = [keypoints[i] for i in idxs]
        descs = [descs[i] for i in idxs]
        
        keypoints = np.asarray(keypoints[: min(max_kps, len(keypoints))])
        descs = np.asarray(descs[: min(max_kps, len(keypoints))])
                
        selection = np.zeros_like(selection)
        for kp in keypoints:
            x, y = kp[0], kp[1]
            selection[int(y)][int(x)] = kp[2]
        
        return selection, keypoints, descs
    
    def edgeFilter(self, img, thresould=10):
        batch = img
        b, c, h, w = batch.size()

        dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)


        dii = F.conv2d(
            batch.view(-1, 1, h, w), dii_filter.to(self.device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            img.view(-1, 1, h, w), dij_filter.to(self.device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), djj_filter.to(self.device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (thresould + 1) ** 2 / thresould
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        return is_not_edge
    
    def nonMaxima(self, img, n=5): # n = window size
        mid = int(n/2)
        h, w = img.shape
        for i in range(h-n-1):
            a = i+n
            for j in range(w-n-1):
                b = j+n
                p = img[mid+i][mid+j]
                if p == 0:
                    continue
                t = False
                for k in range(i, a):
                    for l in range(j, b):
                        if p < img[k][l]:
                            img[mid+i][mid+j] = 0
                            t = True
                            break
                    if t == True:
                        break

        return img