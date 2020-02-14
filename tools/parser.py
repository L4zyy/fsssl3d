import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', '--dataset_root_dir', type=str,
                        help='dataset root directory',
                        default='.' + os.sep + 'datasets' + os.sep + 'modelnet40_images_new_12x')

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
                        help='number of episodes',
                        default=5)

    parser.add_argument('--cuda', action='store_true',
                        help='enable cuda')

    return parser