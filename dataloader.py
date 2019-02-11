import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class dataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        self.batch_table = {4: 512, 8: 512, 16: 512, 32: 256, 64: 256, 128: 128, 256: 128, 512: 64,
                            1024: 32}  # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[pow(2, 2)])  # we start from 2^2=4
        self.imsize = int(pow(2, 2))
        self.num_workers = 4

    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))

        self.batchsize = int(self.batch_table[pow(2, resl)])
        self.imsize = int(pow(2, resl))
        self.dataset = ImageFolder(
            root=self.root,
            transform=transforms.Compose([
                transforms.Resize(size=(self.imsize, self.imsize), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]))

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)  # pixel range [-1, 1]
