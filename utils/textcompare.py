import numpy as np
import torch

def distance(labels, dist_func, num_gpu):
    
    dist = np.zeros(len(labels)-1*num_gpu)
    part = np.linspace(0,len(labels), num_gpu+1, endpoint=True, dtype=np.int32)
    labels = list(labels)

    idx = 0
    for i in range(len(part)-1):
        gpu_section = labels[part[i]:part[i+1]]
        for j in range(len(gpu_section)-1):

            dist[idx] = dist_func(gpu_section[j], gpu_section[j+1])
            idx +=1
    return torch.Tensor(dist) 