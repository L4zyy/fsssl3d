import os
import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultiviewImageDataset(Dataset):
    def __init__(self, root_dir, mode='train', num_model=0, num_views=12, shuffle=True):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.mode = mode
        self.num_views = num_views
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.filepaths = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(root_dir + os.sep +
                                         self.classnames[i] + os.sep +
                                         mode + os.sep +
                                         '*.png'))
            # Select subset for different number of views (12 6 4 3 2 1)
            stride = int(12/self.num_views)
            all_files = all_files[::stride]
            if num_model == 0:
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_model*num_views, len(all_files))])
        
        if shuffle == True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*self.num_views : (rand_idx[i]+1)*self.num_views])
            self.filepaths = filepaths_new
        
        self.y = list(map(lambda path: path.split(os.sep)[-3], self.filepaths[::self.num_views]))
        
    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        imgs = []
        for i in range(self.num_views):
            img = Image.open(self.filepaths[idx*self.num_views + i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        
        return (self.y[idx], torch.stack(imgs), self.filepaths[idx*self.num_views : (idx+1)*self.num_views])
    

if __name__ == "__main__":
    dataset = MultiviewImageDataset('datasets/modelnet40_images_new_12x', mode='train', num_views=12, shuffle=True)

    sample = dataset[0]
    print("[{}] {}".format(sample[0], sample[1].shape))
    print("\n".join(sample[2]))