import cv2, h5py
import os, glob, random
import numpy as np
from torch.utils.data import Dataset
import torch
import gc


SEED = 31
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class NRDataset(Dataset):
    def __init__(self, data_path, batch_size = 8, chunk_size = 10, passes=10, padTo = 0, ori = False):
        self.cnt = 0
        self.chunk_cnt = 0
        self.done = 0
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.passes = passes
        self.chunk_data = None
        self.padTo = padTo
        self.ori = ori

        self.data = h5py.File(data_path, "r")

        self.names = list(self.data['imgs'].keys())
        self.names = list(set(['{:s}__{:s}'.format(*n.split('__')[:2]) for n in self.names]))
        random.shuffle(self.names)

        self.total_len = len(self.names)
        
        self.regularize_names() 
        round_sz = len(self.names) // (batch_size * chunk_size)
        self.names = self.names[:round_sz * batch_size * chunk_size] # trim dataset size to a multiple of batch&chunk  
        print("Total images:", len(self.names))

        self.load_chunk()

    def regularize_names(self):
        dk = {}
        new_names = []
        from collections import deque

        for n in self.names:
            key = n.split('__')[0]
            if key in dk: dk[key].append(n)
            else: dk[key] = deque([n])

        #for v in dk.values():
        #	print(len(v))
                
        done = False
        while not done:
            cnt=0
            for k,v in dk.items():
                if len(dk[k])==0:
                    cnt+=1
                else:
                    new_names.append(dk[k].pop())
            if cnt==len(dk):
                done = True

        self.names = new_names

    def __len__(self):
        return len(self.names) // (self.chunk_size * self.batch_size)

    def load_chunk(self):
        print('loading chunk [{:d}]...'.format(self.chunk_cnt), end='', flush=True)
        self.chunk_data = [] ; gc.collect()
        N = len(self.names) // self.chunk_size
        for i in range(0, N , self.batch_size):
            batch = []
            for b in range(self.batch_size):
                #Read the data from disk
                idx = N * self.chunk_cnt + i + b
                key = self.names[idx]
                H_orig = self.data['imgs/'+key+'__H'][0]
                W_orig = self.data['imgs/'+key+'__W'][0]
                img = self.data['imgs/'+key+'__img'][...]
                y_map = self.data['gts/'+key+'__'][...] # ground truth
                
                # base image
                imgB = self.data['imgs/'+key+'__imgB'][...]
                y_mapB = self.data['gts/'+key+'__B'][...] # ground truth

                y_map = torch.Tensor(y_map)

                y_mapB = torch.Tensor(y_mapB)

                batch.append((y_map, H_orig, W_orig, img, y_mapB, imgB))
            self.chunk_data.append(batch)
        
        self.chunk_cnt+=1
        self.done=0
        if self.chunk_cnt == self.chunk_size:
            self.chunk_cnt = 0

        print('done.')


    def get_raw_batch(self, idx):
        batch = list(zip(*self.chunk_data[idx]))
        return batch

    def __getitem__(self, idx):	
        if self.done == self.passes:
            self.load_chunk()

        batch = list(zip(*self.chunk_data[idx]))
        batch = self.prepare_batch(batch)
        return batch

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt == self.__len__():
            self.done+=1
            self.cnt=0
            raise StopIteration
        else:
            self.cnt+=1
            return self.__getitem__(self.cnt-1)

    def prepare_batch(self, batch):
        Y_img, H, W, img, y_mapB, imgB = batch
        
    
        return Y_img, H, W, img, y_mapB, imgB