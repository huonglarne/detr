from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from data import CocoDetection, collate_fn
from model import Detr


img_folder = "balloon"
train_dataset = CocoDetection(
    img_folder=img_folder + "/train"
)
val_dataset = CocoDetection(
    img_folder=img_folder + "/val", train=False
)

train_dataloader = DataLoader(
    train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True
)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

cats = train_dataset.coco.cats
id2label = {k: v["name"] for k, v in cats.items()}
num_labels = len(id2label)

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=num_labels)


trainer = Trainer(devices=1, max_steps=300, gradient_clip_val=0.1)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
