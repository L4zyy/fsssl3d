import numpy as np
import torch

class PrototypicalBatchSampler(object):
    def __init__(self, labels, class_idxs, num_way, num_support, num_query, num_episode):
        super(PrototypicalBatchSampler, self).__init__()

        self.class_idxs = class_idxs
        self.num_way = num_way
        self.num_sample = num_support + num_query
        self.num_episode = num_episode

        self.classes, self.counts = np.unique(labels, return_counts=True)

        # index table
        self.indices = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        for idx, label in enumerate(labels):
            class_idx = label
            self.indices[class_idx, np.argwhere(np.isnan(self.indices[class_idx]))[0]] = idx
        
        class_idxs = self.class_idxs[torch.randperm(len(self.class_idxs))[:self.num_way]]
    
    def __iter__(self):
        for episode in range(self.num_episode):
            batch_size = self.num_way * self.num_sample
            batch = np.zeros(batch_size, dtype=int)
            class_idxs = self.class_idxs[torch.randperm(len(self.class_idxs))[:self.num_way]]

            for i, c_idx in enumerate(class_idxs):
                c_size = int(self.counts[c_idx])
                s_idxs = torch.randperm(c_size)[:self.num_sample]
                batch[i*self.num_sample : (i+1)*self.num_sample] = self.indices[c_idx][s_idxs]

            yield batch

    def __len__(self):
        return self.num_episode