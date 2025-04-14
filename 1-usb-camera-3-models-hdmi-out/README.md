# 1-usb-camera-3-models-hdmi-out

1 USB camera -> NPU 3 models -> HDMI display (1920 * 1080)

1. Build demo
   ```
   make -j$(nproc)
   ```
2. Check usb camera device name
   ```
   ./check.usbcamera.sh
   ```
3. Run demo
   ```
   Suppose that step 1 gave you /dev/video4
   ./demo -d 4
   ```
