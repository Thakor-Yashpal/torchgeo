# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""ZueriCrop dataset (Time-Series)."""

import os
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.errors import DatasetNotFoundError, RGBBandsMissingError
from torchgeo.datasets.utils import (
    Path,
    download_url,
    lazy_import,
    percentile_normalization,
)


class ZueriCrop(torch.utils.data.Dataset[dict[str, Tensor]]):
    """ZueriCrop dataset (Time-Series).

    Groups the full Sentinel-2 time-series for each field instance
    instead of returning single time steps.

    """


    url = 'https://hf.co/datasets/isaaccorley/zuericrop/resolve/8ac0f416fbaab032d8670cc55f984b9f079e86b2/'
    md5s = ('1635231df67f3d25f4f1e62c98e221a4', '5118398c7a5bbc246f5f6bb35d8d529b')
    filenames = ('ZueriCrop.hdf5', 'labels.csv')

    band_names = ('NIR', 'B03', 'B02', 'B04', 'B05', 'B06', 'B07', 'B11', 'B12')
    rgb_bands = ('B04', 'B03', 'B02')

    def __init__(
        self,
        root: Path = 'data',
        bands: Sequence[str] = band_names,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new ZueriCrop Time-Series dataset instance."""
        lazy_import('h5py')

        self._validate_bands(bands)
        self.band_indices = torch.tensor(
            [self.band_names.index(b) for b in bands]
        ).long()

        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, 'ZueriCrop.hdf5')

        self._verify()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a full time-series for one crop instance."""
        import h5py

        with h5py.File(self.filepath, 'r') as f:
            data = f['data'][index]  # [T, H, W, C]
            mask = f['gt'][index]  # [H, W, C]
            instance_mask = f['gt_instance'][index]

        image = torch.from_numpy(data).permute(0, 3, 1, 2)  # [T, C, H, W]
        image = torch.index_select(image, dim=1, index=self.band_indices)

        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
        instance_tensor = torch.from_numpy(instance_mask).permute(2, 0, 1)

        instance_ids = torch.unique(instance_tensor)
        instance_ids = instance_ids[instance_ids != 0]
        instance_ids = instance_ids[:, None, None]
        masks = instance_tensor == instance_ids

        labels_list = []
        boxes_list = []
        for m in masks:
            label = torch.unique(mask_tensor[m[None, :, :]])[0]
            labels_list.append(label)
            pos = torch.where(m)
            xmin, xmax = torch.min(pos[1]), torch.max(pos[1])
            ymin, ymax = torch.min(pos[0]), torch.max(pos[0])
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.to(torch.uint8)
        boxes = torch.tensor(boxes_list).float()
        labels = torch.tensor(labels_list).long()

        sample: dict[str, Tensor] = {
            'sequence': image,  # [T, C, H, W]
            'mask': masks,
            'bbox_xyxy': boxes,
            'label': labels,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return number of crop instances."""
        import h5py

        with h5py.File(self.filepath, 'r') as f:
            length: int = f['data'].shape[0]
        return length

    def _verify(self) -> None:
        """Verify dataset files exist or download them."""
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            exists.append(os.path.exists(filepath))
        if all(exists):
            return
        if not self.download:
            raise DatasetNotFoundError(self)
        self._download()

    def _download(self) -> None:
        """Download dataset files."""
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                download_url(
                    self.url + filename,
                    self.root,
                    filename=filename,
                    md5=md5 if self.checksum else None,
                )

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate band names."""
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def plot(
        self,
        sample: dict[str, Tensor],
        time_step: int = 0,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset."""
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['sequence'][time_step, rgb_indices]
        mask = torch.argmax(sample['mask'], dim=0)
        image = torch.tensor(
            percentile_normalization(image.numpy()) * 255, dtype=torch.uint8
        )

        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis('off')
        axs[1].imshow(mask)
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
