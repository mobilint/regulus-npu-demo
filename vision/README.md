# SAMPLE CODES FOR VISION MODELS

## 1. Build

Move into each task directory:
```bash
$ cd object-detection
```
Check and update the environment setup path in `compile.sh`:
```bash
source /opt/crosstools/mobilint/1.0.0/v3.2.1/environment-setup-cortexa53-mobilint-linux
```
Build:
```bash
$ ./compile.sh
```

Transfer the built binary to the Regulus board and run it.

## 2. Inference

Move into the build directory:

```bash
$ cd build
```

### 1) Basic execution
```bash
$ ./inference yolov8n-seg.mxq ../sample.jpg
```

`argv[1]`: Path to the `.mxq` model  
(NOTE : The model filename must include one from `["yolo", "ssd"]` and one from `["face", "seg", "pose", ""]`. )

`argv[2]` : Path to the input image

In this case, parameters such as `conf_thres`, `iou_thres`, and `image size` are in `inference.cc`.
The output image will be saved in the same directory as the input image.

### 2) Using a YAML configuration

```bash
$ ./inference_yaml yolov8n-seg.mxq ../model_configs/yolov8n-seg.yaml ../sample.jpg
```

`argv[1]`: Path to the `.mxq` model  

`argv[2]` : Path to the YAML config file

`argv[3]` : Path to the input image

### 3) YAML Configuration

Example
```yaml
Pre-order: [YoloPre]
Pre-processing:
  YoloPre:
    size: [640, 640]
Post-processing:
  type: yolo
  task: object_detection
  nc: 80
  conf_thres: 0.25
  iou_thres: 0.6
  anchors: [
      [12, 16, 19, 36, 40, 28],
      [36, 75, 76, 55, 72, 146],
      [142, 110, 192, 243, 459, 401]
    ]
```

#### Pre-processing ops

YoloPre : `size [h,w]` → letterbox + normalize  
CenterCrop : `size [h,w]`  
Resize :
- `size [h,w]`
- `interpolation` = `bilinear` (default) | `nearest` | `bicubic` | `area`  

Normalize :
- `torch` : `(img - mean) / std`, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- `tf` : `(img / 127.5) - 1`
- `div255` : `img / 255.0`

#### Post-processing

type : `yolo` | `ssd`  
task : `image_classification`, `object_detection`, `instance_segmentation`, `pose_estimation`, `face_detection`  
params :
- nc : num classes  
- nl : num layers  
- conf_thres, iou_thres  
- anchors : optional → if present, anchor-based; if absent, anchor-free


## 3. Tested Models

The following models are tested with the **sample codes** in this repository.  
(Note: These are not the full set of models supported on the Regulus board.)

### Image Classification

| Model | Input Size <br> (H, W, C) |
|------------|------------|
| alexnet | (224, 224, 3) |
| densenet121 | (224, 224, 3) |
| densenet169 | (224, 224, 3) |
| densenet201 | (224, 224, 3) |
| googlenet | (224, 224, 3) |
| mnasnet1_0 | (224, 224, 3) |
| mnasnet1_3 | (224, 224, 3) |
| mobilenet_v2 | (224, 224, 3) |
| regnet_x_400mf | (224, 224, 3) |
| regnet_x_800mf | (224, 224, 3) |
| regnet_x_1_6gf | (224, 224, 3) |
| regnet_x_3_2gf | (224, 224, 3) |
| regnet_x_8gf | (224, 224, 3) |
| regnet_x_16gf | (224, 224, 3) |
| regnet_x_32gf | (224, 224, 3) |
| regnet_y_400mf | (224, 224, 3) |
| regnet_y_800mf | (224, 224, 3) |
| regnet_y_1_6gf | (224, 224, 3) |
| regnet_y_3_2gf | (224, 224, 3) |
| regnet_y_8gf | (224, 224, 3) |
| regnet_y_16gf | (224, 224, 3) |
| regnet_y_32gf | (224, 224, 3) |
| resnet18 | (224, 224, 3) |
| resnet34 | (224, 224, 3) |
| resnet50 | (224, 224, 3) |
| resnet101 | (224, 224, 3) |
| resnet152 | (224, 224, 3) |
| resnext50_32x4d | (224, 224, 3) |
| resnext101_32x8d | (224, 224, 3) |
| resnext101_64x4d | (224, 224, 3) |
| shufflenet_v2_x1_0 | (224, 224, 3) |
| shufflenet_v2_x1_5 | (224, 224, 3) |
| shufflenet_v2_x2_0 | (224, 224, 3) |
| vgg11 | (224, 224, 3) |
| vgg11_bn | (224, 224, 3) |
| vgg13 | (224, 224, 3) |
| vgg13_bn | (224, 224, 3) |
| vgg16 | (224, 224, 3) |
| vgg16_bn | (224, 224, 3) |
| vgg19 | (224, 224, 3) |
| vgg19_bn | (224, 224, 3) |
| wide_resnet50_2 | (224, 224, 3) |
| wide_resnet101_2 | (224, 224, 3) |

### Object Detection

| Model | Input Size <br> (H, W, C) |
|------------|------------|
| ssd_mobilenet_v1 | (300, 300, 3) |
| yolov3 | (640, 640, 3) |
| yolov3u | (640, 640, 3) |
| yolov3-spp | (640, 640, 3) |
| yolov3-sppu | (640, 640, 3) |
| yolov3-tiny | (640, 640, 3) |
| yolov3-tinyu | (640, 640, 3) |
| yolov5n | (640, 640, 3) |
| yolov5s | (640, 640, 3) |
| yolov5m | (640, 640, 3) |
| yolov5l | (640, 640, 3) |
| yolov5x | (640, 640, 3) |
| yolov5nu | (640, 640, 3) |
| yolov5su | (640, 640, 3) |
| yolov5mu | (640, 640, 3) |
| yolov5lu | (640, 640, 3) |
| yolov5xu | (640, 640, 3) |
| yolov5n6 | (1280, 1280, 3) |
| yolov5s6 | (1280, 1280, 3) |
| yolov5m6 | (1280, 1280, 3) |
| yolov5l6 | (1280, 1280, 3) |
| yolov5x6 | (1280, 1280, 3) |
| yolov7 | (640, 640, 3) |
| yolov7x | (640, 640, 3) |
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
| yolov5n-seg | (640, 640, 3) |
| yolov5s-seg | (640, 640, 3) |
| yolov5m-seg | (640, 640, 3) |
| yolov5l-seg | (640, 640, 3) |
| yolov5x-seg | (640, 640, 3) |
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
