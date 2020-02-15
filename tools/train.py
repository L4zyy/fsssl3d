import numpy as np

import torch
from torch.utils.data import DataLoader
from fsssl3d.data.mvi_dataset import MultiviewImageDataset 
from fsssl3d.data.prototypical_batch_sampler import PrototypicalBatchSampler 

from tools.parser import get_parser

def train():
    pass

if __name__ == "__main__":
    args = get_parser().parse_args()

    # load data
    dataset = MultiviewImageDataset(args.dataset_root_dir, mode='test')
    train_sampler=PrototypicalBatchSampler(dataset.y, 'train', args.train_ratio, args.num_way, args.num_support, args.num_query, args.num_episode)
    val_sampler=PrototypicalBatchSampler(dataset.y, 'val', args.train_ratio, args.num_way, args.num_support, args.num_query, args.num_episode)
    train_loader = DataLoader(dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler)

    # check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    print('Using device: ' + device)