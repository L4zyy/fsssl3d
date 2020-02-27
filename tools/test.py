
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

def test(args, test_loader, model, loss_fn):
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    model = model.to(device)

    in_data = None
    out_data = None
    val_acc = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            N, V, C, H, W = data[1].size()
            in_data = Variable(data[1]).view(-1, C, H, W).to(device)
            target = Variable(data[0]).to(device)

            out_data = model(in_data)

            loss, acc = loss_fn(out_data, target)

            val_acc.append(acc.item())
        
    avg_acc = np.mean(val_acc)
    print('test acc: ' + str(avg_acc))


if __name__ == "__main__":
    args = get_parser().parse_args()

    # load data
    test_dataset = MultiviewImageDataset(args.dataset_root_dir, mode='test', num_views=args.num_views)
    test_sampler=PrototypicalBatchSampler(test_dataset.y, 'test', args.train_ratio, args.num_way, args.num_support, args.num_query, args.num_episode)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    # check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    print('Using device: ' + device)

    init_seed(args)

    # single = SVCNN("MVCNN", nclasses=40, pretraining=False, cnn_name="vgg11")
    # model = MVCNN("MVCNN", single, nclasses=40, cnn_name="vgg11", num_views=args.num_views)
    # model.net_2 = model.net_2[:4]
    # del single
    model = SVCNN("MVCNN", nclasses=40, pretraining=False, cnn_name="vgg11")
    model.net_2 = model.net_2[:4]

    best_model_path = os.path.join(args.result_root_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test(args, test_loader, model, PrototypicalLoss(args.num_support))
