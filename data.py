import os
import torchvision
from transformers import DetrFeatureExtractor

FEATURE_EXTRACTOR = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, train=True):
        ann_file = os.path.join(
            img_folder, "custom_train.json" if train else "custom_val.json"
        )
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = FEATURE_EXTRACTOR

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = FEATURE_EXTRACTOR.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
