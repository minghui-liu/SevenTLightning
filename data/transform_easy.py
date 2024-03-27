from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    SpatialPadd,
    CenterSpatialCropd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

def get_train_transforms(args):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if args.spacing:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=args.spacing_pixdim,
                mode=args.spacing_mode,
            )
        )
    if args.scale_intensity:
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.scale_intensity_range[0],
                a_max=args.scale_intensity_range[1],
                b_min=args.scale_intensity_target[0],
                b_max=args.scale_intensity_target[1],
                clip=args.scale_intensity_clip,
            )
        )
    if args.spatial_pad:
        transforms.append(
            SpatialPadd(keys=["image", "label"], spatial_size=args.img_size)
        )
    if args.center_spatial_crop:
        transforms.append(
            CenterSpatialCropd(keys=["image", "label"], roi_size=args.img_size)
        )
    if args.crop_foreground:
        transforms.append(
            CropForegroundd(keys=["image", "label"], source_key="image")
        )
    if args.rand_crop_by_pos_neg_label:
        transforms.append(
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=args.rand_crop_by_pos_neg_label_spatial_size,
                pos=args.rand_crop_by_pos_neg_label_pos,
                neg=args.rand_crop_by_pos_neg_label_neg,
                num_samples=args.rand_crop_by_pos_neg_label_num_samples,
                image_key="image",
                image_threshold=args.rand_crop_by_pos_neg_label_image_threshold,
            )
        )
    if args.flip0:
        transforms.append(
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=args.flip_prob,
            )
        )
    if args.flip1:
        transforms.append(
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=args.flip_prob,
            )
        )
    if args.flip2:
        transforms.append(
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=args.flip_prob,
            )
        )
    # if rotate90:
    #     transforms.append(
    #         RandRotate90d(
    #             keys=["image", "label"],
    #             prob=rotate90_prob,
    #             max_k=3,
    #         )
    #     )
    if args.shift_intensity:
        transforms.append(
            RandShiftIntensityd(
                keys=["image"],
                offsets=args.shift_intensity_offset,
                prob=args.shift_intensity_prob,
            )
        )
    train_transforms = Compose(transforms)
    return train_transforms

def get_val_transforms(args):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if args.spacing:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=args.spacing_pixdim,
                mode=args.spacing_mode,
            )
        )
    if args.scale_intensity:
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.scale_intensity_range[0],
                a_max=args.scale_intensity_range[1],
                b_min=args.scale_intensity_target[0],
                b_max=args.scale_intensity_target[1],
                clip=args.scale_intensity_clip,
            )
        )
    if args.crop_foreground:
        transforms.append(
            CropForegroundd(keys=["image", "label"], source_key="image")
        )
    val_transforms = Compose(transforms)
    return val_transforms
