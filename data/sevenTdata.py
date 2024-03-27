import lightning as L
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

class SevenTDataModule(L.LightningDataModule):
    def __init__(self, args,
                 train_transform=None, val_transform=None,
                 ):
        super().__init__()
        self.args = args
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_files = load_decathlon_datalist(self.args.datalist_json, True, "training")
            self.train_ds = CacheDataset(self.train_files, transform=self.train_transform, cache_num=self.args.train_cache_num)
            self.val_files = load_decathlon_datalist(self.args.datalist_json, True, "validation")
            self.val_ds = CacheDataset(self.val_files, transform=self.val_transform, cache_num=self.args.val_cache_num)

        if stage == 'validate':
            self.val_files = load_decathlon_datalist(self.data_list, True, "validation")
            self.val_ds = CacheDataset(self.val_files, transform=self.val_transform, cache_num=self.args.val_cache_num)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False)
