from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from attacks import AttackSpec


class NpzBackdoorDataset(Dataset):
    def __init__(self, numpy_file, trigger_file, transform=None, trigger_size=None, trigger_target=0):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array["x"]
        self.targets = torch.Tensor(self.input_array["y"])
        trigger_arrays = np.load(trigger_file)
        self.trigger_patch_list = trigger_arrays["t"]
        self.trigger_mask_list = trigger_arrays["tm"]
        self.test_transform = transform
        self.trigger_size = trigger_size
        self.trigger_target = int(trigger_target)

    def __getitem__(self, index):
        img_backdoor = self.data[index].copy()
        img = self.data[index].copy()
        if self.trigger_size:
            ori_size = img_backdoor.shape
            img_backdoor = cv2.resize(img_backdoor, (self.trigger_size, self.trigger_size))
            img = cv2.resize(img, (self.trigger_size, self.trigger_size))
        img_backdoor[:] = img_backdoor * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        if self.trigger_size:
            img_backdoor = cv2.resize(img_backdoor, (ori_size[1], ori_size[1]))
        img_backdoor = self.test_transform(Image.fromarray(img_backdoor))
        img = self.test_transform(Image.fromarray(img))
        label = self.targets[index]
        label_trigger = torch.tensor(self.trigger_target)
        return img_backdoor, img, label, label_trigger

    def __len__(self):
        return self.data.shape[0]


class DRUPEImageNetBackdoor(Dataset):
    def __init__(self, base_dataset, indices, trigger_file, transform=None, trigger_size=None, trigger_target=0):
        self.base_dataset = base_dataset
        self.indices = indices
        self.test_transform = transform or base_dataset.transform
        self.trigger_size = trigger_size
        trigger_arrays = np.load(trigger_file)
        self.trigger_patch_list = trigger_arrays["t"]
        self.trigger_mask_list = trigger_arrays["tm"]
        self.trigger_target = int(trigger_target)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image_rel_path = self.base_dataset.images.iloc[base_idx]
        image_path = Path(self.base_dataset.root) / image_rel_path
        img = np.array(Image.open(image_path).convert("RGB"))
        img_backdoor = img.copy()
        if self.trigger_size:
            ori_size = img_backdoor.shape
            img_backdoor = cv2.resize(img_backdoor, (self.trigger_size, self.trigger_size))
            img = cv2.resize(img, (self.trigger_size, self.trigger_size))
        img_backdoor[:] = img_backdoor * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        if self.trigger_size:
            img_backdoor = cv2.resize(img_backdoor, (ori_size[1], ori_size[0]))
        img_backdoor = self.test_transform(Image.fromarray(img_backdoor.astype(np.uint8)))
        img = self.test_transform(Image.fromarray(img.astype(np.uint8)))
        label = torch.tensor(int(self.base_dataset.labels.iloc[base_idx]), dtype=torch.long)
        label_trigger = torch.tensor(self.trigger_target, dtype=torch.long)
        return img_backdoor, img, label, label_trigger


@dataclass
class DatasetSpec:
    name: str
    trusted_loader: DataLoader
    mixed_loader: DataLoader
    pair_loader: bool


def build_npz_pair_dataset(
    data_dir: str,
    trigger_path: str,
    batch_size: int,
    num_workers: int,
    trigger_size: int = 32,
    transform=None,
) -> DatasetSpec:
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )

    train_dataset = NpzBackdoorDataset(
        numpy_file=str(Path(data_dir) / "train.npz"),
        trigger_file=trigger_path,
        transform=transform,
        trigger_size=trigger_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DatasetSpec(
        name="npz_pair",
        trusted_loader=train_loader,
        mixed_loader=train_loader,
        pair_loader=True,
    )


def build_badencoder_cifar10_pair(
    data_dir: str,
    trigger_path: str,
    batch_size: int,
    num_workers: int,
    trigger_size: int = 32,
) -> DatasetSpec:
    from BadEncoder.datasets.cifar10_dataset import test_transform_cifar10

    return build_npz_pair_dataset(
        data_dir=data_dir,
        trigger_path=trigger_path,
        batch_size=batch_size,
        num_workers=num_workers,
        trigger_size=trigger_size,
        transform=test_transform_cifar10,
    )


def build_imagenet_subset_pair(
    imagenet_val_dir: str,
    labels_csv: str,
    trigger_path: str,
    processor,
    batch_size: int,
    num_workers: int,
    clean_subset_frac: float = 0.1,
) -> DatasetSpec:
    from BadCLIP.src.data import ImageLabelDataset
    from types import SimpleNamespace

    common_opts = dict(
        eval_test_data_csv=str(labels_csv),
        add_backdoor=False,
        backdoor_sufi=False,
        label="airplane",
        patch_size=16,
        patch_type="ours_tnature",
        patch_location="middle",
        tigger_pth=None,
        patch_name=None,
        blended_alpha=None,
        scale=None,
        save_files_name=None,
        eval_test_data_dir=str(imagenet_val_dir),
    )
    clean_options = SimpleNamespace(**common_opts)
    clean_dataset = ImageLabelDataset(
        root=str(imagenet_val_dir),
        transform=processor.process_image,
        options=clean_options,
    )

    clean_sample_size = max(1, int(len(clean_dataset) * float(clean_subset_frac)))
    generator = torch.Generator().manual_seed(42)
    clean_indices = torch.randperm(len(clean_dataset), generator=generator)[:clean_sample_size].tolist()
    clean_subset = Subset(clean_dataset, clean_indices)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    clean_loader = DataLoader(clean_subset, **loader_kwargs)

    trigger_dataset = DRUPEImageNetBackdoor(
        base_dataset=clean_dataset,
        indices=clean_indices,
        trigger_file=trigger_path,
        transform=processor.process_image,
        trigger_size=224,
        trigger_target=0,
    )
    mixed_loader = DataLoader(trigger_dataset, **loader_kwargs)

    return DatasetSpec(
        name="imagenet_subset_pair",
        trusted_loader=clean_loader,
        mixed_loader=mixed_loader,
        pair_loader=True,
    )


def build_cifar10_224_clean(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    trusted_frac: float = 0.1,
    augment: bool = True,
) -> DatasetSpec:
    from DeDe.new_defense.dataset_transforms.cifar10 import Cifar10TrainingDataset, Cifar10ValDataset

    data_dir = Path(data_dir)
    train_npz = data_dir / "train_224.npz"
    test_npz = data_dir / "test_224.npz"
    train_data = np.load(str(train_npz))
    test_data = np.load(str(test_npz))
    train_x, train_y = train_data["x"], train_data["y"]
    test_x, test_y = test_data["x"], test_data["y"]

    if trusted_frac < 1.0:
        idx = np.random.choice(len(train_x), size=int(trusted_frac * len(train_x)), replace=False)
        train_x = train_x[idx]
        train_y = train_y[idx]

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )

    train_dataset = Cifar10TrainingDataset(train_x, train_y, transform=train_transform)
    test_dataset = Cifar10ValDataset(test_x, test_y)

    trusted_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    mixed_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DatasetSpec(
        name="cifar10_224_clean",
        trusted_loader=trusted_loader,
        mixed_loader=mixed_loader,
        pair_loader=False,
    )


def build_dataset_from_args(args, attack_spec: AttackSpec) -> DatasetSpec:
    if args.dataset == "cifar10_npz_pair":
        return build_npz_pair_dataset(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
        )
    if args.dataset == "badencoder_cifar10_pair":
        return build_badencoder_cifar10_pair(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
        )
    if args.dataset == "imagenet_subset_pair":
        if attack_spec.processor is None:
            raise ValueError("imagenet_subset_pair requires a CLIP-compatible attack with processor")
        return build_imagenet_subset_pair(
            imagenet_val_dir=args.imagenet_val_dir,
            labels_csv=args.labels_csv,
            trigger_path=args.trigger_path,
            processor=attack_spec.processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clean_subset_frac=args.clean_subset_frac,
        )
    if args.dataset == "cifar10_224_clean":
        return build_cifar10_224_clean(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trusted_frac=args.trusted_frac,
            augment=not args.no_augment,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


# Local import to avoid hard dependency at module import time
import cv2  # noqa: E402