import numpy as np

import torch
from torch.utils.data import DataLoader
from fsssl3d.data.mvi_dataset import MultiviewImageDataset 
from fsssl3d.data.prototypical_batch_sampler import PrototypicalBatchSampler 

def train(dataset_root_dir, num_way, num_support, num_query, num_episode, gpu):
    # load data
    dataset = MultiviewImageDataset(dataset_root_dir, mode='train')
    sampler=PrototypicalBatchSampler(dataset.y, num_way, num_support, num_query, num_episode)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'
    print('Using device: ' + device)

if __name__ == "__main__":
    train('./datasets/modelnet40_images_new_12x', 5, 5, 5, 200, gpu=True)