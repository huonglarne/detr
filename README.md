# Get data

    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip

    unzip balloon_dataset.zip

    git clone https://github.com/woctezuma/VIA2COCO
    cd VIA2COCO/
    git checkout fixes

    python preprocess_data.py

# Run with moreh

    conda create -n detr python=3.8

    update-moreh --target 23.3.0 --force --torch 1.11

    pip install -r requirements.txt

    python finetune_object_detection.py



