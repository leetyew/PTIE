import torch
import csv
import os
import PIL
import h5py
import numpy as np
import torchvision
from scipy.io import loadmat
from PIL import Image
import traceback
import random
import re
import pickle
import data
from io import BytesIO
import copy
import base64
import math
import cv2
from transforms import CVColorJitter, CVDeterioration, CVGeometry
import lmdb
import six

# TODO

# Clean up 

   
def generate_corpus(x, all_upper = False):
    idx = []
    #default_idx = [ord (i) for i in list('abcdefghijklmnopqrstuvqxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')]
    for i in x:
        idx.extend(list(i.upper() if all_upper else i))
    idx = [ord(i) for i in idx]
    #idx = idx + default_idx
    idx = list(set(idx))
    
    idx2char = {0 : chr(65533), 1 : chr(9218), 2 : chr(9219)}
    char2idx = {chr(65533): 0, chr(9218) : 1, chr(9219) : 2}
    
    for i in range(len(idx)):
        idx2char.update({i+3 : chr(idx[i])})
        char2idx.update({chr(idx[i]) : i+3})

    corpus = {'idx2char': idx2char, 'char2idx': char2idx}
    return corpus   

def word2tensor(corpus, txt, all_upper = False):
    w = []
    for word in txt:
        word = word.upper() if all_upper else word
        temp = []
        for c in word:
            try:
                temp.append(corpus['char2idx'][c])
            except KeyError:
                temp.append(0)
                
        w.append(torch.LongTensor([1, *temp, 2]))
    return w

def word2tensor_gt(corpus, txt, all_upper = False):
    w = []
    for word in txt:
        word = word.upper() if all_upper else word
        temp = []
        for c in word:
            try:
                temp.append(corpus['char2idx'][c])
            except KeyError:
                temp.append(0)
                
        w.append(torch.LongTensor([*temp, 2]))
    return w

def word2tensor_combined(corpus, txt, all_upper = False):
    w = []
    w_gt = []
    for word in txt:
        word = word.upper() if all_upper else word
        temp = []
        for c in word:
            try:
                temp.append(corpus['char2idx'][c])
            except KeyError:
                temp.append(0)
                
        w.append(torch.LongTensor([1, *temp, 2]))
        w_gt.append(torch.LongTensor([*temp, 2]))
    return w, w_gt

def word2tensor_CTC(corpus, txt, all_upper = False):
    w = []
    for word in txt:
        word = word.upper() if all_upper else word
        temp = []
        for c in word:
            try:
                temp.append(corpus['char2idx'][c])
            except KeyError:
                temp.append(0)
                
        w.append(torch.LongTensor([*temp]))
    return w

def tensor2word_CTC(corpus, tensors, im_len):
    w = []
    for i in range(len(tensors)):
        temp = ''
        tensor = tensors[i][:im_len[i]]
        for idx in tensor:
            if idx == -1:
                break
            else:
                temp += (corpus['idx2char'][idx])
        w.append(temp)
    return w

def tensor2word(corpus, tensors):
    w = []
    for tensor in tensors:
        temp = ''
        for idx in tensor:
            if idx == 2:
                break
            else:
                temp += (corpus['idx2char'][idx])
        w.append(temp)
    return w

def apply_padding(seq, max_seq_len, pad=0):
    batch_size = len(seq)
    seq_out = torch.ones([batch_size, max_seq_len], dtype=torch.long) * pad
    for i in range(batch_size):
        seq_out[i, :len(seq[i])] = seq[i]
    
    return seq_out

class MjSynthv1_SynthTextv1_Dataset(torch.utils.data.Dataset):
    # TODO:

    def __init__(self, transform, dataset_type, **kwargs):
        
        with open('../ocr_T/data/meta_ST_MJ.txt', 'r') as f:
            self.meta = f.readlines()
            self.meta = [line.rstrip('\n') for line in self.meta]

        self.transform = transform
        self.width = kwargs['im_width']
        self.height = kwargs['im_height']
        self.transform_resize = torchvision.transforms.Compose([torchvision.transforms.Resize((self.height,self.width))])
        self.op = kwargs['op']
        self.augment = kwargs['augment']
        if self.augment:
            self.augment_tfs = torchvision.transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.as_augment = kwargs['as_augment']
        self.pad = kwargs['pad']
        self.direction = kwargs['direction']
        self.multiscale = kwargs['multiscale']
                                       
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        try:
            meta_split = self.meta[idx].split(', ',1)
            text = meta_split[1]
            assert len(text) <= 28
            if self.direction == 'RL':
                text = text[::-1]
                
            img_name = '../ocr_T/data/' + meta_split[0]
 
            if '.jpg' in img_name:
                image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            else:
                image = np.load(img_name)


            if self.op == 'ViT_dualPatches':
                image = Image.fromarray(image)
                if self.augment:
                    image = self.augment_tfs(image)
                image = self.transform_resize(image)
                image_4by8 = torchvision.transforms.ToTensor()(image).squeeze().permute(1,0).reshape(128,8,4).permute(1,0,2).reshape(-1,32)
                image_8by4 = torchvision.transforms.ToTensor()(image).squeeze().permute(1,0).reshape(128,4,8).permute(1,0,2).reshape(-1,32)                
                if self.direction == 'bidirectional':
                    return image_4by8, image_8by4, text, text[::-1]
                else:
                    return image_4by8, image_8by4, text
            
        except Exception as e:
            if str(e) == "module 'PIL' has no attribute 'Image'":
                print(e)
            #print(e)
            import traceback
            traceback.print_exc()
            return None


class SynthDataset_Val_ViT(torch.utils.data.Dataset):
    def __init__(self, transform, **kwargs):
        self.transform = transform
        self.data_type = kwargs['dataset_type']
        self.op = kwargs['op']
        self.file = None
        self.as_pad = kwargs['as_pad']
        file = h5py.File('../ocr_T/data/hdf5/SynFull_Val_32by128_unpadded.h5', 'r')
        self.len = file['/images'].shape[0]
        self.pad = kwargs['pad']
        self.direction = kwargs['direction']
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.file = h5py.File('../ocr_T/data/hdf5/SynFull_Val_32by128_unpadded.h5', 'r')

        image = self.file["/images"][idx]
        text = self.file["/labels"][idx]
                
        text = str(text, 'utf-8')
        if self.direction == 'RL':
            text = text[::-1]
            
        if self.op == 'ViT_dualPatches':
            img_resized_4by8 = torchvision.transforms.ToTensor()(image).squeeze().permute(1,0).reshape(128,8,4).permute(1,0,2).reshape(-1,32)
            img_resized_8by4 = torchvision.transforms.ToTensor()(image).squeeze().permute(1,0).reshape(128,4,8).permute(1,0,2).reshape(-1,32)
            if self.direction == 'bidirectional':
                return img_resized_4by8, img_resized_8by4, text, text[::-1]
            else:
                return img_resized_4by8, img_resized_8by4, text
            
class cute80Dataset_Custom(torch.utils.data.Dataset):
    # TODO:

    def __init__(self, transform, dataset_type, **kwargs):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        if dataset_type == 'Train':
            csv_file=''
            root_dir=''
        elif dataset_type == 'Test':
            csv_file='../ocr_T/data/cute80/gt.txt'
            root_dir='../ocr_T/data/cute80/'
        else:
            raise ValueError('No such dataset type: ' + str(dataset_type))
            
        self.meta = [] # list of lists
        with open(csv_file, 'r') as file:
            reader = file.read().splitlines()
            for row in reader:
                row_0 = os.path.join(root_dir, row.split(' ')[0])
                row_1 = row.split(' ')[1]
                self.meta.append((row_0, row_1))   
                    
        self.root_dir = root_dir
        self.transform = transform
        self.width = kwargs['im_width']
        self.height = kwargs['im_height']
        self.transform_resize = torchvision.transforms.Compose([torchvision.transforms.Resize((self.height,self.width))])
        self.op = kwargs['op']
        self.direction = kwargs['direction']

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.meta[idx][0]
        if self.out_channel == 1:
            image = PIL.Image.open(img_name).convert("L")
        elif self.out_channel == 3:
            image = PIL.Image.open(img_name)
        text = self.meta[idx][1]
         
        if self.direction == 'RL':
            text = text[::-1]    
                
        w, h = image.size        
        if self.pad == True:
            w, h = image.size
            size = (self.height, w*self.height//h)
            image_resized = torchvision.transforms.functional.resize(image, size)
            w, h = image_resized.size
            if w > self.width:
                image = np.array(self.transform_resize(image))
                p = False
            else:
                image = np.zeros((self.height,self.width),dtype='uint8')
                image[:self.height, :w] = np.array(image_resized)
                p = True
        else:
            image = np.array(self.transform_resize(image))
            
        if self.op == 'ViT_dualPatches':
            img_resized_4by8 = torchvision.transforms.ToTensor()(image).squeeze().permute(1,0).reshape(128,8,4).permute(1,0,2).reshape(-1,32)
            img_resized_8by4 = torchvision.transforms.ToTensor()(image).squeeze().permute(1,0).reshape(128,4,8).permute(1,0,2).reshape(-1,32)
            if self.direction == 'bidirectional':
                return img_resized_4by8, img_resized_8by4, text, text[::-1]
            else:
                return img_resized_4by8, img_resized_8by4, text
