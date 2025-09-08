# SAMPLE CODES FOR VISION MODELS

## 1. Build

Move into the each task directory:
```
$ cd object-detection
```
Check and update the environment setup path in `compile.sh`:
```
source /opt/crosstools/mobilint/1.0.0/v3.2.1/environment-setup-cortexa53-mobilint-linux
```
Build:
```
$ ./compile.sh
```

Transfer the built binary to the regulus board and run it.

## 2. Inference

Move into the build directory:

```
$ cd build
```

### 1) Basic execution (hard-coded parameters)
```
$ ./inference yolov8n-seg.mxq ../sample.jpg
```

`argv[1]`: Path to the `.mxq` model  
(NOTE : The model filename must include one from `["yolo", "ssd"]` and one from `["face", "seg", "pose", ""]`. )

`argv[2]` : Path to the input image

In this case, parameters such as `conf_thres`, `iou_thres`, and `image size` are in inference.cc. You can adjust them.
The output image will be saved in the same directory as the input image.

### 2) using yaml config

```
$ ./inference_yaml yolov8n-seg.mxq ../model_configs/yolov9c-seg.yaml ../sample.jpg
```

`argv[1]`: mxq model_path 

`argv[2]` : model config yaml file path

`argv[3]` : img file for inference


## 3. supported models

### Object Detection

| Model | Input Size <br> (H, W, C) |
|------------|------------|
| ssd_mobilenet_v1 | (300, 300, 3) |
| yolov8n | (640, 640, 3) |
| yolov8s | (640, 640, 3) |
| yolov8m | (640, 640, 3) |
| yolov8l | (640, 640, 3) |
| yolov8x | (640, 640, 3) |
| yolov9t | (640, 640, 3) |
| yolov9s | (640, 640, 3) |
| yolov9m | (640, 640, 3) |
| yolov9c | (640, 640, 3) |

### Instance Segmentation

| Model | Input Size <br> (H, W, C) |
|------------|------------|
| yolov8n-seg | (640, 640, 3) |
| yolov8s-seg | (640, 640, 3) |
| yolov8m-seg | (640, 640, 3) |
| yolov8l-seg | (640, 640, 3) |
| yolov8x-seg | (640, 640, 3) |
| yolov9c-seg | (640, 640, 3) |

### Pose Estimation

| Model | Input Size <br> (H, W, C) |
|------------|------------|
| yolov8n-pose | (640, 640, 3) |
| yolov8s-pose | (640, 640, 3) |
| yolov8m-pose | (640, 640, 3) |
| yolov8l-pose | (640, 640, 3) |

### Face Detection

| Model | Input Size <br> (H, W, C) |
|------------|------------|
| [yolov8n-face](https://github.com/derronqi/yolov8-face) | (640, 640, 3) |