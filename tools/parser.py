import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', type=str,
                        help='name for this run',
                        default='demo')

    parser.add_argument('-root', '--dataset_root_dir', type=str,
                        help='dataset root directory',
                        default='.' + os.sep + 'datasets' + os.sep + 'modelnet40_images_new_12x')

    parser.add_argument('-result', '--result_root_dir', type=str,
                        help='result root directory',
                        default='.' + os.sep + 'result')

    parser.add_argument('-v', '--num_views', type=int,
                        help='number of views',
                        default=12)

    parser.add_argument('-w', '--num_way', type=int,
                        help='number of ways',
                        default=5)

    parser.add_argument('-s', '--num_support', type=int,
                        help='number of support samples',
                        default=5)

    parser.add_argument('-q', '--num_query', type=int,
                        help='number of query samples',
                        default=5)

    parser.add_argument('-e', '--num_episode', type=int,
                        help='number of episodes per epoch',
                        default=100)

    parser.add_argument('-epoch', '--num_epoches', type=int,
                        help='number of epoches',
                        default=100)

    parser.add_argument('--train_size', type=int,
                        help='number of class for train',
                        default=24)

    parser.add_argument('--val_size', type=int,
                        help='number of class for validation',
                        default=6)

    parser.add_argument('--cuda', action='store_true',
                        help='enable cuda')

    parser.add_argument('--seed', type=int,
                        help='random seed',
                        default=1)


    return parser