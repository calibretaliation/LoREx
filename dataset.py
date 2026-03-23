"""Dataset builders for LoREx.

There are two independent concerns handled here:

1. **Trusted dataset** (``build_trusted_loader``): clean images used to
   estimate the whitening matrix.  This must be completely independent of
   the attack's training/test data.  Supports CIFAR-10, SVHN, GTSRB, STL-10,
   ImageNet (ImageFolder), arbitrary NPZ files, and arbitrary ImageFolder trees.

2. **Test dataset** (``build_*_pair`` / ``build_*_clean`` builders): the
   images actually fed to the detector — some clean, some poisoned.  These are
   attack-specific and returned bundled as a ``DatasetSpec``.

``build_dataset_from_args()`` assembles both pieces from CLI args.
If ``--trusted_dataset`` is given, a separate trusted loader is constructed
and injected into the returned ``DatasetSpec``, overriding the embedded one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision.transforms import transforms

from attacks import AttackSpec


# ── Per-dataset normalization constants (32×32 models) ──────────────────────
# Use these as the default transform when attack_spec.transform is not set.
DATASET_NORM = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "svhn":    ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
    "gtsrb":   ([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
    "stl10":   ([0.4467, 0.4398, 0.4066], [0.2242, 0.2215, 0.2239]),
}


def _default_npz_transform(dataset_name: str) -> transforms.Compose:
    """Minimal normalize-only transform for 32×32 NPZ datasets."""
    mean, std = DATASET_NORM.get(dataset_name, DATASET_NORM["cifar10"])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class NpzCleanDataset(Dataset):
    """Clean NPZ dataset — no trigger applied.  Returns (image_tensor, label).

    NPZ files must contain:
      ``x``: uint8 array of shape (n, H, W, 3)
      ``y``: integer label array of shape (n,)  [optional; defaults to zeros]
    """

    def __init__(self, npz_path: str, transform: Optional[Callable] = None):
        data = np.load(npz_path)
        self.data = data["x"]
        self.labels = data["y"].astype(np.int64) if "y" in data else np.zeros(
            len(self.data), dtype=np.int64
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        img = self.transform(img) if self.transform else img
        return img, int(self.labels[idx])


class NpzBackdoorDataset(Dataset):
    """Paired (poisoned, clean) NPZ dataset.

    Returns 4-tuples: (poisoned_img, clean_img, clean_label, trigger_target).
    Used for pair-loader test sets (DRUPE/BadEncoder on CIFAR-10, SVHN, etc.).
    """

    def __init__(self, numpy_file, trigger_file, transform=None,
                 trigger_size=None, trigger_target=0):
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
        img_backdoor[:] = (
            img_backdoor * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        )
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
    def __init__(self, base_dataset, indices, trigger_file, transform=None,
                 trigger_size=None, trigger_target=0):
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
        img_backdoor[:] = (
            img_backdoor * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        )
        if self.trigger_size:
            img_backdoor = cv2.resize(img_backdoor, (ori_size[1], ori_size[0]))
        img_backdoor = self.test_transform(Image.fromarray(img_backdoor.astype(np.uint8)))
        img = self.test_transform(Image.fromarray(img.astype(np.uint8)))
        label = torch.tensor(int(self.base_dataset.labels.iloc[base_idx]), dtype=torch.long)
        label_trigger = torch.tensor(self.trigger_target, dtype=torch.long)
        return img_backdoor, img, label, label_trigger


class BadCLIPImageNetPair(Dataset):
    """Pair dataset for BadCLIP using its native apply_trigger() function."""

    def __init__(self, base_dataset, indices, transform, patch_type, patch_location,
                 patch_size, patch_name, trigger_target=0, scale=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.test_transform = transform
        self.patch_type = patch_type
        self.patch_location = patch_location
        self.patch_size = patch_size
        self.patch_name = patch_name
        self.scale = scale
        self.trigger_target = int(trigger_target)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        from BadCLIP.backdoor.utils import apply_trigger
        from types import SimpleNamespace

        base_idx = self.indices[idx]
        image_rel_path = self.base_dataset.images.iloc[base_idx]
        image_path = Path(self.base_dataset.root) / image_rel_path
        img_pil = Image.open(image_path).convert("RGB")

        args = SimpleNamespace(
            patch_size=self.patch_size,
            patch_name=self.patch_name,
            scale=self.scale,
        )
        img_backdoor_pil = apply_trigger(
            img_pil.copy(),
            patch_size=self.patch_size,
            patch_type=self.patch_type,
            patch_location=self.patch_location,
            args=args,
        )

        img = self.test_transform(img_pil)
        img_backdoor = self.test_transform(img_backdoor_pil)
        label = torch.tensor(int(self.base_dataset.labels.iloc[base_idx]), dtype=torch.long)
        label_trigger = torch.tensor(self.trigger_target, dtype=torch.long)
        return img_backdoor, img, label, label_trigger


# ─────────────────────────────────────────────────────────────────────────────
# DatasetSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """Bundles the two loaders needed by LoREx's detection pipeline.

    trusted_loader: DataLoader over CLEAN images used to fit the whitening
        matrix.  Should be independent of the attack's training/test data.
    mixed_loader:   DataLoader over TEST images.  If pair_loader=True, yields
        (poisoned, clean, label, trigger_label) 4-tuples.  If pair_loader=False,
        yields (clean, label) 2-tuples and LoREx applies the trigger on-the-fly
        via the attack's trigger_model.
    pair_loader:    True  → extract_clean_and_poison_features_pair()
                    False → extract_clean_and_poison_features_trigger()
    """
    name: str
    trusted_loader: DataLoader
    mixed_loader: DataLoader
    pair_loader: bool


# ─────────────────────────────────────────────────────────────────────────────
# Trusted dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_trusted_loader(
    trusted_dataset: str,
    data_dir: str,
    transform: Optional[Callable],
    n_trusted: int,
    batch_size: int,
    num_workers: int,
    use_224: bool = False,
    split: str = "train",
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader over a clean trusted dataset.

    This is the sole source of images used to estimate the whitening matrix.
    It is always clean (no backdoor trigger applied) and should be independent
    of the attack's training and test data.

    Args:
        trusted_dataset: One of "cifar10", "svhn", "gtsrb", "stl10",
            "imagenet", "custom_npz", "custom_folder".
        data_dir: For named datasets (cifar10/svhn/gtsrb/stl10): the directory
            that contains the NPZ files (e.g. ``/data/svhn/``).
            For "imagenet": root of an ImageFolder tree.
            For "custom_npz": exact path to the .npz file.
            For "custom_folder": root of an ImageFolder tree.
        transform: PIL-image → Tensor transform matching the encoder's expected
            input format.  Use ``attack_spec.transform``.  If None, falls back
            to a dataset-appropriate normalize-only transform.
        n_trusted: Number of images to sample for the trusted set.
        batch_size: Loader batch size.
        num_workers: DataLoader workers.
        use_224: For NPZ datasets, load the ``<split>_224.npz`` variant
            (pre-resized to 224×224).  Required for CLIP-based encoders.
        split: "train" or "test" — which NPZ split to load.
        seed: Random seed for the subset sample.

    Returns:
        DataLoader that yields (image_tensor, label) batches.
    """
    NPZ_DATASETS = {"cifar10", "svhn", "gtsrb", "stl10"}

    if trusted_dataset in NPZ_DATASETS:
        suffix = f"_{split}_224.npz" if use_224 else f"_{split}.npz"
        # Normalise: user may pass ".../svhn" or ".../svhn/train.npz"
        p = Path(data_dir)
        if p.suffix == ".npz":
            npz_path = str(p)
        else:
            npz_path = str(p / (split + ("_224" if use_224 else "") + ".npz"))

        if transform is None:
            transform = _default_npz_transform(trusted_dataset)

        dataset = NpzCleanDataset(npz_path, transform=transform)

    elif trusted_dataset == "imagenet":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        dataset = tv_datasets.ImageFolder(data_dir, transform=transform)

    elif trusted_dataset == "custom_npz":
        if transform is None:
            raise ValueError("--trusted_dataset custom_npz requires a model transform "
                             "(set via attack_spec.transform or provide --trusted_dataset "
                             "with a named dataset)")
        dataset = NpzCleanDataset(data_dir, transform=transform)

    elif trusted_dataset == "custom_folder":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        dataset = tv_datasets.ImageFolder(data_dir, transform=transform)

    else:
        raise ValueError(
            f"Unknown trusted_dataset: {trusted_dataset!r}. "
            "Choose from cifar10/svhn/gtsrb/stl10/imagenet/custom_npz/custom_folder."
        )

    # Subsample n_trusted images
    n = len(dataset)
    n_sample = min(n_trusted, n)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:n_sample].tolist()
    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def build_npz_pair_dataset(
    data_dir: str,
    trigger_path: str,
    batch_size: int,
    num_workers: int,
    trigger_size: int = 32,
    transform=None,
    dataset_name: str = "cifar10",
) -> DatasetSpec:
    """Pair-loader for NPZ backdoor datasets (CIFAR-10, SVHN, GTSRB, STL-10).

    The trusted_loader and mixed_loader both iterate the same underlying NPZ
    file.  ``extract_features(trusted_loader)`` takes ``batch[0]`` which is
    the CLEAN image (index 1 of the pair), so the whitening matrix is clean.

    Note: if you want a fully independent trusted set, pass ``--trusted_dataset``
    in the CLI so ``build_trusted_loader()`` overrides this loader.
    """
    if transform is None:
        transform = _default_npz_transform(dataset_name)

    train_npz = str(Path(data_dir) / "train.npz")
    pair_dataset = NpzBackdoorDataset(
        numpy_file=train_npz,
        trigger_file=trigger_path,
        transform=transform,
        trigger_size=trigger_size,
    )
    # Use a clean-only subset as the trusted loader (takes batch[1] = clean img)
    clean_dataset = NpzCleanDataset(train_npz, transform=transform)

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    trusted_loader = DataLoader(clean_dataset, **loader_kwargs)
    mixed_loader = DataLoader(pair_dataset, **loader_kwargs)

    return DatasetSpec(
        name=f"{dataset_name}_npz_pair",
        trusted_loader=trusted_loader,
        mixed_loader=mixed_loader,
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
        dataset_name="cifar10",
    )


def build_svhn_npz_pair(
    data_dir: str,
    trigger_path: str,
    batch_size: int,
    num_workers: int,
    trigger_size: int = 32,
    transform=None,
) -> DatasetSpec:
    """Pair-loader for SVHN NPZ with backdoor trigger."""
    return build_npz_pair_dataset(
        data_dir=data_dir,
        trigger_path=trigger_path,
        batch_size=batch_size,
        num_workers=num_workers,
        trigger_size=trigger_size,
        transform=transform,
        dataset_name="svhn",
    )


def build_gtsrb_npz_pair(
    data_dir: str,
    trigger_path: str,
    batch_size: int,
    num_workers: int,
    trigger_size: int = 32,
    transform=None,
) -> DatasetSpec:
    """Pair-loader for GTSRB NPZ with backdoor trigger."""
    return build_npz_pair_dataset(
        data_dir=data_dir,
        trigger_path=trigger_path,
        batch_size=batch_size,
        num_workers=num_workers,
        trigger_size=trigger_size,
        transform=transform,
        dataset_name="gtsrb",
    )


def build_stl10_npz_pair(
    data_dir: str,
    trigger_path: str,
    batch_size: int,
    num_workers: int,
    trigger_size: int = 32,
    transform=None,
) -> DatasetSpec:
    """Pair-loader for STL-10 NPZ with backdoor trigger."""
    return build_npz_pair_dataset(
        data_dir=data_dir,
        trigger_path=trigger_path,
        batch_size=batch_size,
        num_workers=num_workers,
        trigger_size=trigger_size,
        transform=transform,
        dataset_name="stl10",
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
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False, shuffle=False,
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


def build_badclip_imagenet_pair(
    imagenet_val_dir: str,
    labels_csv: str,
    processor,
    batch_size: int,
    num_workers: int,
    patch_type: str,
    patch_location: str,
    patch_size: int,
    patch_name: str,
    clean_subset_frac: float = 0.1,
    scale=None,
) -> DatasetSpec:
    from BadCLIP.src.data import ImageLabelDataset
    from types import SimpleNamespace

    common_opts = dict(
        eval_test_data_csv=str(labels_csv),
        add_backdoor=False,
        backdoor_sufi=False,
        label="airplane",
        patch_size=patch_size,
        patch_type=patch_type,
        patch_location=patch_location,
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
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False, shuffle=False,
    )
    clean_loader = DataLoader(clean_subset, **loader_kwargs)

    trigger_dataset = BadCLIPImageNetPair(
        base_dataset=clean_dataset,
        indices=clean_indices,
        transform=processor.process_image,
        patch_type=patch_type,
        patch_location=patch_location,
        patch_size=patch_size,
        patch_name=patch_name,
        scale=scale,
    )
    mixed_loader = DataLoader(trigger_dataset, **loader_kwargs)

    return DatasetSpec(
        name="badclip_imagenet_pair",
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
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

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


def build_imagenet_clean(
    imagenet_val_dir: str,
    labels_csv: str,
    processor,
    batch_size: int,
    num_workers: int,
    trusted_frac: float = 0.1,
) -> DatasetSpec:
    """Clean ImageNet dataset for CLIP-based encoders (non-pair).

    Splits ImageNet validation into a trusted clean fraction and a mixed
    fraction.  pair_loader=False — triggers are applied on-the-fly by the
    attack's trigger model (e.g. INACTIVE UNet).
    """
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
    options = SimpleNamespace(**common_opts)
    full_dataset = ImageLabelDataset(
        root=str(imagenet_val_dir),
        transform=processor.process_image,
        options=options,
    )

    n = len(full_dataset)
    n_trusted = max(1, int(n * trusted_frac))
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=generator)
    trusted_indices = perm[:n_trusted].tolist()
    mixed_indices = perm[n_trusted:].tolist()

    trusted_subset = Subset(full_dataset, trusted_indices)
    mixed_subset = Subset(full_dataset, mixed_indices)

    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False, shuffle=False,
    )
    trusted_loader = DataLoader(trusted_subset, **loader_kwargs)
    mixed_loader = DataLoader(mixed_subset, **loader_kwargs)

    return DatasetSpec(
        name="imagenet_clean",
        trusted_loader=trusted_loader,
        mixed_loader=mixed_loader,
        pair_loader=False,
    )


def build_custom_folder_clean(
    folder_path: str,
    transform: Callable,
    batch_size: int,
    num_workers: int,
    trusted_frac: float = 0.1,
    seed: int = 42,
) -> DatasetSpec:
    """Clean ImageFolder dataset for custom directories (non-pair).

    Expects the standard ImageFolder layout::

        folder_path/
            class_a/img1.jpg
            class_b/img2.jpg
            ...

    pair_loader=False — triggers applied on-the-fly by the attack's trigger_model.
    """
    full_dataset = tv_datasets.ImageFolder(folder_path, transform=transform)

    n = len(full_dataset)
    n_trusted = max(1, int(n * trusted_frac))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    trusted_indices = perm[:n_trusted].tolist()
    mixed_indices = perm[n_trusted:].tolist()

    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False, shuffle=False,
    )
    trusted_loader = DataLoader(Subset(full_dataset, trusted_indices), **loader_kwargs)
    mixed_loader = DataLoader(Subset(full_dataset, mixed_indices), **loader_kwargs)

    return DatasetSpec(
        name="custom_folder_clean",
        trusted_loader=trusted_loader,
        mixed_loader=mixed_loader,
        pair_loader=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_from_args(
    args,
    attack_spec: AttackSpec,
    trusted_loader: Optional[DataLoader] = None,
) -> DatasetSpec:
    """Build a DatasetSpec from CLI args.

    Args:
        args: parsed argparse namespace.
        attack_spec: AttackSpec (used for processor / transform when needed).
        trusted_loader: if provided, overrides the ``trusted_loader`` embedded
            in the dataset builder.  Pass the result of ``build_trusted_loader``
            when ``--trusted_dataset`` is set.

    Returns:
        DatasetSpec with trusted_loader and mixed_loader ready to use.
    """
    if args.dataset == "cifar10_npz_pair":
        spec = build_npz_pair_dataset(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
            dataset_name="cifar10",
        )
    elif args.dataset == "badencoder_cifar10_pair":
        spec = build_badencoder_cifar10_pair(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
        )
    elif args.dataset == "svhn_npz_pair":
        spec = build_svhn_npz_pair(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
            transform=attack_spec.transform,
        )
    elif args.dataset == "gtsrb_npz_pair":
        spec = build_gtsrb_npz_pair(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
            transform=attack_spec.transform,
        )
    elif args.dataset == "stl10_npz_pair":
        spec = build_stl10_npz_pair(
            data_dir=args.data_dir,
            trigger_path=args.trigger_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trigger_size=args.trigger_size,
            transform=attack_spec.transform,
        )
    elif args.dataset == "imagenet_subset_pair":
        if attack_spec.processor is None:
            raise ValueError("imagenet_subset_pair requires a CLIP-compatible attack with processor")
        spec = build_imagenet_subset_pair(
            imagenet_val_dir=args.imagenet_val_dir,
            labels_csv=args.labels_csv,
            trigger_path=args.trigger_path,
            processor=attack_spec.processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clean_subset_frac=args.clean_subset_frac,
        )
    elif args.dataset == "badclip_imagenet_pair":
        if attack_spec.processor is None:
            raise ValueError("badclip_imagenet_pair requires a CLIP-compatible attack with processor")
        spec = build_badclip_imagenet_pair(
            imagenet_val_dir=args.imagenet_val_dir,
            labels_csv=args.labels_csv,
            processor=attack_spec.processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patch_type=args.patch_type,
            patch_location=args.patch_location,
            patch_size=args.patch_size,
            patch_name=args.patch_name,
            clean_subset_frac=args.clean_subset_frac,
        )
    elif args.dataset == "cifar10_224_clean":
        spec = build_cifar10_224_clean(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trusted_frac=args.trusted_frac,
            augment=not args.no_augment,
        )
    elif args.dataset == "imagenet_clean":
        if attack_spec.processor is None:
            raise ValueError("imagenet_clean requires a CLIP-compatible attack with processor")
        spec = build_imagenet_clean(
            imagenet_val_dir=args.imagenet_val_dir,
            labels_csv=args.labels_csv,
            processor=attack_spec.processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trusted_frac=args.trusted_frac,
        )
    elif args.dataset == "custom_folder_clean":
        if attack_spec.transform is None:
            raise ValueError(
                "custom_folder_clean requires a model transform. "
                "The attack's transform is not set — this is unexpected."
            )
        spec = build_custom_folder_clean(
            folder_path=args.data_dir,
            transform=attack_spec.transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trusted_frac=args.trusted_frac,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset!r}")

    # If a separately-built trusted loader was provided, override the embedded one.
    # This is the case when --trusted_dataset is specified in the CLI.
    if trusted_loader is not None:
        spec = DatasetSpec(
            name=spec.name,
            trusted_loader=trusted_loader,
            mixed_loader=spec.mixed_loader,
            pair_loader=spec.pair_loader,
        )

    return spec


# Local import to avoid hard dependency at module import time
import cv2  # noqa: E402
