from data.image import *

import numpy as np
import torch


def make_collate_fn(randomize_n_views=False, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        batch['keypoints'] = [item['keypoints'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['action'] = [item['action'] for item in items]
        batch['frame'] = [item['frame'] for item in items]

        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
