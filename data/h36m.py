from data import data_utils as d_utils
from cfg import constants as _C

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from collections import defaultdict
from copy import copy, deepcopy
import pickle
import os; import os.path as osp


class Human36M(Dataset):
    def __init__(self, root_pth, label_pth,
                 **kwargs
                 ):

        super(Human36M, self).__init__()

        self.root_pth = root_pth
        self.labels = np.load(osp.join(root_pth, label_pth), allow_pickle=True).item()

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']

        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)

        indices = []
        mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
        indices.append(np.nonzero(mask)[0])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]

    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, idx):
        sample = defaultdict(list)
        shot = self.labels['table'][idx]
        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']

        sample['keypoints'] = shot['keypoints'][_C.J32_TO_J17]

        # save sample's index
        sample['indexes'] = idx
        sample['dataset'] = 'Human36M'
        sample['action'] = action
        sample['frame'] = frame_idx
        sample.default_factory = None

        return sample


def setup_human36m_dloader(args, **kwargs):

    print('Load H36M Dataset...')

    train_dataset = Human36M(root_pth=args.root_pth,
                             label_pth=args.label_pth,
                             train=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  sampler=None,
                                  collate_fn=d_utils.make_collate_fn(),
                                  num_workers=args.num_workers,
                                  worker_init_fn=d_utils.worker_init_fn,
                                  pin_memory=True)

    return train_dataloader
