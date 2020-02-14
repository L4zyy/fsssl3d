import numpy as np

from torch.utils.data import DataLoader
from fsssl3d.data.mvi_dataset import MultiviewImageDataset 
from fsssl3d.data.prototypical_batch_sampler import PrototypicalBatchSampler 

def train(dataset_root_dir, num_way, num_support, num_query, num_episode):
    # load data
    dataset = MultiviewImageDataset(dataset_root_dir, mode='train')
    sampler=PrototypicalBatchSampler(dataset.y, num_way, num_support, num_query, num_episode)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

if __name__ == "__main__":
    train('./datasets/modelnet40_images_new_12x', 5, 5, 5, 200)