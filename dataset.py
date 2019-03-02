import torch
import torch.utils.data as data


class Sprites(data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = torch.load(self.path+'/%d.sprite' % idx)
        return item['id'], item['body'], item['bottom'], item['top'], item['hair'], item['action'], item['orientation'], item['frames']
