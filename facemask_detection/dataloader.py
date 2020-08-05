from pathlib import Path
from typing import List, Tuple, Dict, Any

import albumentations as albu
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class FaceMaskDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform: albu.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, target = self.samples[idx]

        image = load_rgb(image_path, lib="cv2")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {
            "image_id": image_path.stem,
            "image": tensor_from_rgb_image(image),
            "target": torch.Tensor([target]),
        }


class FaceMaskTestDataset(Dataset):
    def __init__(self, samples: List[Path], transform: albu.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.samples[idx]

        image = load_rgb(image_path, lib="cv2")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {"image_id": image_path.stem, "image": tensor_from_rgb_image(image), "image_path": str(image_path)}
