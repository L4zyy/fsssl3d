import numpy as np
import torch

class PrototypicalBatchSampler(object):
    def __init__(self, labels, num_way, num_support, num_query, num_episode):
        super(PrototypicalBatchSampler, self).__init__()

        self.num_way = num_way
        self.num_sample = num_support + num_query
        self.num_episode = num_episode

        self.classes, self.counts = np.unique(labels, return_counts=True)

        # index table
        self.indices = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        for idx, label in enumerate(labels):
            class_idx = np.argwhere(self.classes == label).item()
            self.indices[class_idx, np.argwhere(np.isnan(self.indices[class_idx]))[0]] = idx
    
    def __iter__(self):
        for episode in range(self.num_episode):
            batch_size = self.num_way * self.num_sample
            batch = np.zeros(batch_size, dtype=int)
            class_idxs = torch.randperm(len(self.classes))[:self.num_way]

            for i, c_idx in enumerate(class_idxs):
                s_idxs = torch.randperm(int(self.counts[c_idx]))[:self.num_sample]
                batch[i*self.num_sample : (i+1)*self.num_sample] = self.indices[c_idx][s_idxs]

            yield batch

    def __len__(self):
        return self.num_episode