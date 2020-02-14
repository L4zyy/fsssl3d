import numpy as np

import torch
from torch.utils.data import DataLoader
from fsssl3d.data.mvi_dataset import MultiviewImageDataset 
from fsssl3d.data.prototypical_batch_sampler import PrototypicalBatchSampler 

from tools.parser import get_parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    # load data
    dataset = MultiviewImageDataset(args.dataset_root_dir, mode='train')
    sampler=PrototypicalBatchSampler(dataset.y, args.num_way, args.num_support, args.num_query, args.num_episode)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    print('Using device: ' + device)