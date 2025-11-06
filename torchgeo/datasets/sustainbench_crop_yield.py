# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""SustainBench Crop Yield dataset (Time-Series)."""

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .errors import DatasetNotFoundError
from .utils import Path, download_url, extract_archive


class SustainBenchCropYield(Dataset):
    """SustainBench Crop Yield Dataset (Time-Series).

    Groups samples across years for each location/county and returns a
    full temporal sequence per region instead of individual samples.

    Each item in the dataset is a dictionary:
        {
            "sequence": Tensor [T, C, H, W],
            "years": Tensor [T],
            "labels": Tensor [T],
            "ndvi": Tensor [T, ...],
            "meta": { "region": str, "country": str }
        }
    """

    valid_countries = ('usa', 'brazil', 'argentina')

    md5 = '362bad07b51a1264172b8376b39d1fc9'

    url = 'https://drive.google.com/file/d/1lhbmICpmNuOBlaErywgiD6i9nHuhuv0A/view?usp=drive_link'

    dir = 'soybeans'

    valid_splits = ('train', 'dev', 'test')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        countries: list[str] = ['usa'],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance."""
        assert set(countries).issubset(self.valid_countries), (
            f'Please choose a subset of these valid countries: {self.valid_countries}.'
        )
        self.countries = countries

        assert split in self.valid_splits, (
            f'Please choose one of these valid data splits {self.valid_splits}.'
        )
        self.split = split

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        # region -> list of yearly samples
        self.groups = {}

        for country in self.countries:
            image_file_path = os.path.join(
                self.root, self.dir, country, f'{self.split}_hists.npz'
            )
            target_file_path = image_file_path.replace('_hists', '_yields')
            years_file_path = image_file_path.replace('_hists', '_years')
            ndvi_file_path = image_file_path.replace('_hists', '_ndvi')

            npz_file = np.load(image_file_path)['data']
            target_npz_file = np.load(target_file_path)['data']
            year_npz_file = np.load(years_file_path)['data']
            ndvi_npz_file = np.load(ndvi_file_path)['data']
            num_data_points = npz_file.shape[0]
            for idx in range(num_data_points):
                image = torch.from_numpy(npz_file[idx]).permute(2, 0, 1).to(torch.float32)
                label = float(target_npz_file[idx])
                year = int(year_npz_file[idx])
                ndvi = torch.from_numpy(ndvi_npz_file[idx]).to(dtype=torch.float32)

                region_id = f"{country}_{idx % 1000}"

                entry = {
                    'image': image,
                    'label': label,
                    'year': year,
                    'ndvi': ndvi,
                    'country': country,
                }
                self.groups.setdefault(region_id, []).append(entry)

        for region in self.groups:
            self.groups[region].sort(key=lambda e: e['year'])

        self.keys = list(self.groups.keys())

    def __len__(self) -> int:
        """Return the number of regions in the dataset."""
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return the full time-series sample for a given region."""
        region_id = self.keys[index]
        entries = self.groups[region_id]
        sequence = torch.stack([e['image'] for e in entries])
        years = torch.tensor([e['year'] for e in entries], dtype=torch.int32)
        labels = torch.tensor([e['label'] for e in entries], dtype=torch.float32)
        ndvi = torch.stack([e['ndvi'] for e in entries])
        sample: dict[str, Tensor] = {
            'sequence': sequence,
            'years': years,
            'labels': labels,
            'ndvi': ndvi,
            'meta': {'region': region_id, 'country': entries[0]['country']},
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample

    def _verify(self) -> None:
        pathname = os.path.join(self.root, self.dir)
        if os.path.exists(pathname):
            return

        pathname = os.path.join(self.root, self.dir) + '.zip'
        if os.path.exists(pathname):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        download_url(
            self.url,
            self.root,
            filename=self.dir + '.zip',
            md5=self.md5 if self.checksum else None,
        )
        self._extract()

    def _extract(self) -> None:
        zipfile_path = os.path.join(self.root, self.dir) + '.zip'
        extract_archive(zipfile_path, self.root)
