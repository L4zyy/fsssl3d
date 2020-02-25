import numpy as np
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import matplotlib.pyplot as plt
import tqdm

from fsssl3d.data.mvi_dataset import MultiviewImageDataset 
from fsssl3d.data.prototypical_batch_sampler import PrototypicalBatchSampler 
from fsssl3d.network.mvcnn.MVCNN import SVCNN, MVCNN
from fsssl3d.utils.fsssl3d_loss import PrototypicalLoss

from tools.parser import get_parser

def init_seed(args):
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, train_loader, val_loader, model, optim, loss_fn):
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    # writer = SummaryWriter(args.result_root_dir)

    best_state = None
    best_acc = 0
    i_acc = 0

    best_model_path = os.path.join(args.result_root_dir, 'best_model.pth')
    last_model_path = os.path.join(args.result_root_dir, 'last_model.pth')

    model = model.to(device)
    model.train()
    for epoch in range(args.num_episode):
        # plot learning rate
        # lr = optim.state_dict()["param_groups"][0]['lr']
        # writer.add_scalar('params/lr', lr, epoch)

        in_data = None
        out_data = None
        for i, data in enumerate(train_loader):
            N, V, C, H, W = data[1].size()
            in_data = Variable(data[1]).view(-1, C, H, W).to(device)
            target = Variable(data[0]).to(device)

            optim.zero_grad()

            out_data = model(in_data)
            print(out_data.shape)

            loss = loss_fn(out_data, target)

            return



if __name__ == "__main__":
    args = get_parser().parse_args()

    # load data
    train_dataset = MultiviewImageDataset(args.dataset_root_dir, mode='train')
    test_dataset = MultiviewImageDataset(args.dataset_root_dir, mode='test')
    train_sampler=PrototypicalBatchSampler(train_dataset.y, 'train', args.train_ratio, args.num_way, args.num_support, args.num_query, args.num_episode)
    val_sampler=PrototypicalBatchSampler(train_dataset.y, 'val', args.train_ratio, args.num_way, args.num_support, args.num_query, args.num_episode)
    test_sampler=PrototypicalBatchSampler(train_dataset.y, 'test', args.train_ratio, args.num_way, args.num_support, args.num_query, args.num_episode)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    # check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    print('Using device: ' + device)

    init_seed(args)

    single = SVCNN("MVCNN", nclasses=40, pretraining=True, cnn_name="vgg11")
    model = MVCNN("MVCNN", single, nclasses=40, cnn_name="vgg11", num_views=12)
    del single

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.001, betas=(0.9, 0.999))

    result = train(args, train_loader, val_loader, model, optimizer, PrototypicalLoss)
    # best_state, best_acc, train_loss, train_acc, val_loss, val_acc = result
