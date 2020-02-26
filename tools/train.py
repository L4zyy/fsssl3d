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
from fsssl3d.utils.prototypical_loss import PrototypicalLoss

from tools.parser import get_parser

def init_seed(args):
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, train_loader, val_loader, model, optim, loss_fn):
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    writer = SummaryWriter(args.result_root_dir)

    best_state = None
    best_acc = 0
    i_acc = 0

    best_model_path = os.path.join(args.result_root_dir, 'best_model.pth')
    last_model_path = os.path.join(args.result_root_dir, 'last_model.pth')

    model = model.to(device)
    model.train()
    for epoch in range(args.num_epoches):
        # plot learning rate
        lr = optim.state_dict()["param_groups"][0]['lr']
        writer.add_scalar('params/lr', lr, epoch)

        in_data = None
        out_data = None
        for i, data in enumerate(train_loader):
            N, V, C, H, W = data[1].size()
            in_data = Variable(data[1]).view(-1, C, H, W).to(device)
            target = Variable(data[0]).to(device)

            optim.zero_grad()

            out_data = model(in_data)

            loss, acc = loss_fn(out_data, target)

            loss.backward()
            optim.step()

            writer.add_scalar('train/train_loss', loss, i_acc+i+1)
            writer.add_scalar('train/train_acc', acc, i_acc+i+1)
        
        i_acc += i

        # evaluation
        with torch.no_grad():
            val_acc = []
            for i, data in enumerate(val_loader):
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).to(device)
                target = Variable(data[0]).to(device)

                out_data = model(in_data)

                loss, acc = loss_fn(out_data, target)

                writer.add_scalar('val/val_loss', loss, i_acc+i+1)
                writer.add_scalar('val/val_acc', acc, i_acc+i+1)
                val_acc.append(acc.item())
        
        # select best model
        avg_acc = np.mean(val_acc)
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        print('finish epoch: ' + str(epoch) + ', acc: ' + str(avg_acc))
    # save last model
    torch.save(model.state_dict(), last_model_path)

    return best_state



if __name__ == "__main__":
    args = get_parser().parse_args()

    # load data
    train_dataset = MultiviewImageDataset(args.dataset_root_dir, mode='train', num_views=args.num_views)
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

    single = SVCNN("MVCNN", nclasses=40, pretraining=False, cnn_name="vgg11")
    model = MVCNN("MVCNN", single, nclasses=40, cnn_name="vgg11", num_views=args.num_views)
    model.net_2 = model.net_2[:4]
    del single

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.001, betas=(0.9, 0.999))

    result = train(args, train_loader, val_loader, model, optimizer, PrototypicalLoss(args.num_support))
