# SAMPLE CODES FOR VISION MODELS

## 1. build

`compile.sh` 의 environment 버전 및 경로 확인
```
source /opt/crosstools/mobilint/1.0.0/v3.2.1/environment-setup-cortexa53-mobilint-linux
```
```
$ ./compile.sh
```

ssh 등을 통해 빌드된 binary 파일을 regulus로 전송해 실행

## 2. Inference

```
$ cd build
```
```
$ ./inference yolov9c-seg.mxq ../sample.jpg
```

`argv[1]`: mxq model_path (NOTE : model name must contain task - one of `["yolo", "ssd", "seg", "pose"]` )

`argv[2]` : img file for inference

The inference result image will be saved in the same directory as the source image.


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