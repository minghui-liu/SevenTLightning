import torch
import datetime
import monai
import argparse
import os
import time
import lightning as L

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from sklearn.metrics import confusion_matrix
from data.transform_easy import get_train_transforms, get_val_transforms
from data.sevenTdata import SevenTDataModule
from models.unetr import UNETR

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-op", "--optimizer", type=str, default="AdamW")
    parser.add_argument("-m", "--momentum", type=float, default=0)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0)
    parser.add_argument("-d", "--dropout_rate", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--datalist_json", type=str, default="data/file_set.json")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("-e", "--epochs", type=int, default=400)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=10)

    # data loading arguments
    parser.add_argument("-w", "--num_workers", type=int, default=2)
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--train_cache_num", type=int, default=16)
    parser.add_argument("--val_cache_num", type=int, default=8)
    parser.add_argument("--network_path", type=str, default="")
    parser.add_argument("--local_path", type=str, default="")

    # model arguments
    parser.add_argument("--model", type=str, default="unetr")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--spatial_dims", type=int, default=3)
    parser.add_argument("--norm_name", type=str, default="instance")

    # unetr arguments
    parser.add_argument("--window_size", type=tuple, default=(128, 128, 128))
    parser.add_argument("--img_size", type=tuple, default=(288, 448, 448))
    parser.add_argument("--feature_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--mlp_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--pos_embed", type=str, default="perceptron")
    parser.add_argument("--conv_block", type=bool, default=True)
    parser.add_argument("--res_block", type=bool, default=True)
    parser.add_argument("--qkv_bias", type=bool, default=False)
    parser.add_argument("--save_attn", type=bool, default=False)

    # unet arguments
    parser.add_argument("--channels", type=tuple, default=(4, 8, 16, 32, 64))
    parser.add_argument("--strides", type=tuple, default=(2, 2, 2, 2))
    parser.add_argument("--kernel_size", type=tuple, default=(3, 3, 3))
    parser.add_argument("--up_kernel_size", type=tuple, default=(3, 3, 3))
    parser.add_argument("--num_res_units", type=int, default=0)
    parser.add_argument("--act", type=str, default="prelu")
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--adn_ordering", type=str, default="NDA")

    # augmentation arguments
    parser.add_argument("--spacing", type=bool, default=True)
    parser.add_argument("--spacing_mode", type=tuple, default=("bilinear", "nearest"))
    parser.add_argument("--spacing_pixdim", type=tuple, default=(0.5, 0.5, 0.5))
    parser.add_argument("--scale_intensity", type=bool, default=True)
    parser.add_argument("--scale_intensity_range", type=tuple, default=(-175, 25000))
    parser.add_argument("--scale_intensity_target", type=tuple, default=(0.0, 1.0))
    parser.add_argument("--scale_intensity_clip", type=bool, default=True)
    parser.add_argument("--crop_foreground", type=bool, default=False)
    parser.add_argument("--spatial_pad", type=bool, default=False)
    parser.add_argument("--center_spatial_crop", type=bool, default=False)
    parser.add_argument("--rand_crop_by_pos_neg_label", type=bool, default=True)
    parser.add_argument("--rand_crop_by_pos_neg_label_spatial_size", type=tuple, default=(128, 128, 128))
    parser.add_argument("--rand_crop_by_pos_neg_label_pos", type=float, default=1.0)
    parser.add_argument("--rand_crop_by_pos_neg_label_neg", type=float, default=0.0)
    parser.add_argument("--rand_crop_by_pos_neg_label_num_samples", type=int, default=1)
    parser.add_argument("--rand_crop_by_pos_neg_label_image_threshold", type=float, default=0.0)
    parser.add_argument("--flip0", type=bool, default=True)
    parser.add_argument("--flip1", type=bool, default=True)
    parser.add_argument("--flip2", type=bool, default=True)
    parser.add_argument("--flip_prob", type=float, default=0.1)
    parser.add_argument("--rotate90", type=bool, default=True)
    parser.add_argument("--rotate90_prob", type=float, default=0.1)
    parser.add_argument("--shift_intensity", type=bool, default=True)
    parser.add_argument("--shift_intensity_offset", type=float, default=0.1)
    parser.add_argument("--shift_intensity_prob", type=float, default=0.1)

    # lightning
    parser.add_argument("--devices", type=int, default=1)

    args = parser.parse_args()


# log all arguments to stdout
print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

train_transforms = get_train_transforms(args)
val_transforms = get_val_transforms(args)

data_module = SevenTDataModule(args, train_transform=train_transforms, val_transform=val_transforms)

class UNetR_Lightning(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.args = args

        self.post_label = AsDiscrete(to_onehot=args.out_channels)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    def configure_model(self):
        self.model = UNETR(
            in_channels=self.args.in_channels,
            out_channels=self.args.out_channels,
            # img_size=self.args.img_size,
            img_size=self.args.window_size,
            feature_size=self.args.feature_size,
            hidden_size=self.args.hidden_size,
            mlp_dim=self.args.mlp_dim,
            num_heads=self.args.num_heads,
            pos_embed=self.args.pos_embed,
            norm_name=self.args.norm_name,
            res_block=self.args.res_block,
            dropout_rate=self.args.dropout_rate,
            conv_block=self.args.conv_block,
            spatial_dims=self.args.spatial_dims,
            qkv_bias=self.args.qkv_bias,
            save_attn=self.args.save_attn
        )

     def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["label"]
        logit_map = self.model(images)
        loss = self.loss(logit_map, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["label"]
        output = sliding_window_inference(images, self.args.window_size, self.args.batch_size, self.model)

        mask_list = decollate_batch(masks)
        mask_convert = [self.post_label(m) for m in mask_list]
        output_list = decollate_batch(output)
        output_convert = [self.post_pred(o) for o in output_list]
        self.metric(y_pred=output_convert, y=mask_convert)
        mean_dice_val = self.metric.aggregate().item()
        self.metric.reset()
        self.log("val_dice", mean_dice_val)


if args.model == "unetr":
    model_lightning = UNetR_Lightning(args)
else:
    raise ValueError(f"invalid model value {args.model}, must be unetr or unet")

trainer = L.Trainer(min_epochs=50, max_epochs=args.epochs, accelerator="cuda", devices=args.devices, strategy="fsdp")
trainer.fit(model_lightning, datamodule=data_module)

