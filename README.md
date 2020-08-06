# Facemask detection
Detection masks on faces.

## Train set

### No mask:
*  [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
*  [WiderFace](http://shuoyang1213.me/WIDERFACE/)

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
