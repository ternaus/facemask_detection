# Facemask detection

It could be confusing, but the model in this library perform classifications of the images.
It takes image as an input and outputs probability of person in the image wearing a mask.

Hence in order to get expected results the model should be combined with face detector, for example from
https://github.com/ternaus/retinaface.

[Example on how to combine face detector with mask detector](https://colab.research.google.com/drive/13Ktsrx164eQHfDmYLyMCoI-Kq0gC5Kg1?usp=sharing)


![https://habrastorage.org/webt/b_/ja/ww/b_jawwxndpkdl2pjlxlcxvars6m.png](https://habrastorage.org/webt/b_/ja/ww/b_jawwxndpkdl2pjlxlcxvars6m.png)

# Use
```python
import albumentations as A
import torch
from facemask_detection.pre_trained_models import get_model

model = get_model("tf_efficientnet_b0_ns_2020-07-29")
model.eval()

transform = A.Compose([A.SmallestMaxSize(max_size=256, p=1),
                       A.CenterCrop(height=224, width=224, p=1),
                       A.Normalize(p=1)])
```
`image = <numpy array with the shape (height, width, 3)>`

```python

transformed_image = transform(image=image)['image']

input = torch.from_numpy(np.transpose(transformed_image, (2, 0, 1))).unsqueeze(0)

print("Probability of the mask on the face = ", model(input)[0].item())
```

* Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VkSK5MKIuGPIA31KJpGiFe_FafYC4xfD?usp=sharing)
* Jupyter notebook with the example on how to combine face detector with mask detector: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13Ktsrx164eQHfDmYLyMCoI-Kq0gC5Kg1?usp=sharing)
## Train set

Train dataset was composed from the data:

### No mask:
*  [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

### Mask:
*  [https://www.kaggle.com/andrewmvd/face-mask-detection](https://www.kaggle.com/andrewmvd/face-mask-detection)
*  [https://www.kaggle.com/alexandralorenzo/maskdetection](https://www.kaggle.com/alexandralorenzo/maskdetection)
*  [https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
*  [https://humansintheloop.org/medical-mask-dataset](https://humansintheloop.org/medical-mask-dataset)


# Trainining

Define config, similar to [facemask_detection_configs/2020-07-29.yaml](facemask_detection_configs/2020-07-29.yaml).

Run

```bash
python facemask_detection/train.py -c <config>
```

Inference

```bash
python -m torch.distributed.launch --nproc_per_node=1 facemask_detection/inference.py -h
usage: inference.py [-h] -i INPUT_PATH -c CONFIG_PATH -o OUTPUT_PATH
                    [-b BATCH_SIZE] [-j NUM_WORKERS] -w WEIGHT_PATH
                    [--world_size WORLD_SIZE] [--local_rank LOCAL_RANK]
                    [--fp16]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path with images.
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path to config.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save jsons.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size
  -j NUM_WORKERS, --num_workers NUM_WORKERS
                        num_workers
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        Path to weights.
  --world_size WORLD_SIZE
                        number of nodes for distributed training
  --local_rank LOCAL_RANK
                        node rank for distributed training
  --fp16                Use fp6
```

Example:

```
python -m torch.distributed.launch --nproc_per_node=<num_gpu> facemask_detection/inference.py \
                                   -i <input_path> \
                                   -w <path to weights> \
                                   -o <path to the output_csv> \
                                   -c <path to config>
                                   -b <batch size>
```
