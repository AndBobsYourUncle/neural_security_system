# Neural Security System

The goal of this tool is to allow monitoring of security cameras, and to actively publish the presense of humans to an MQTT topic. MQTT is a pubsub type protocol. This was inspired from [https://github.com/PINTO0309/OpenVINO-YoloV3](https://github.com/PINTO0309/OpenVINO-YoloV3)

# Running via Docker

This whole app has recently been packaged up into a Docker image for convenience. If you'd like to see live output from the app, you will still have to follow the instructions below for building it within a Ubuntu 18.04 LTS Desktop environment.

Using the CPU:
```bash
docker run  \
  -v /home/nicholas/cameras.yaml:/usr/neural_security_system/cameras.yaml \
  -e CAMERAS="cameras.yaml" \
  -e MODEL="./models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.xml" \
  -e DEVICE="CPU" \
  -e MQTT_USER="MQTT_USER" -e MQTT_PASSWORD="MQTT_PASSWORD" \
  -e MQTT_HOST="tcp://MQTT_HOST_IP:1883" andbobsyouruncle/neural_security_system
```

And then like this if you happen to have an Intel Neural Compute Stick 2 plugged into the host machine running docker:
```bash
docker run --privileged -v /dev/bus/usb:/dev/bus/usb \
  -v /home/nicholas/cameras.yaml:/usr/neural_security_system/cameras.yaml \
  -e CAMERAS="cameras.yaml" \
  -e MODEL="./models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.xml" \
  -e DEVICE="MYRIAD" \
  -e MQTT_USER="MQTT_USER" -e MQTT_PASSWORD="MQTT_PASSWORD" \
  -e MQTT_HOST="tcp://MQTT_HOST_IP:1883" andbobsyouruncle/neural_security_system
```

You must have a cameras.yaml file on the host machine, and it should look something like this:
```
cameras:
  - name: Front Door
    input: http://192.168.1.52:8081
    mqtt_topic: cameras/front_door/humans
    crop_top: 80
    crop_right: 150
    crop_bottom: 0
    crop_left: 0
  - name: Driveway
    input: http://192.168.1.52:8082
    mqtt_topic: cameras/driveway/humans
    crop_top: 0
    crop_right: 0
    crop_bottom: 0
    crop_left: 0
```

Use the path to this file whever you save it to on the host machine for the `-v /home/nicholas/cameras.yaml:/usr/neural_security_system/cameras.yaml` portion of `docker run`.

Also, if you’d like to mess with some of the other neural models provided in the image, you have all these available:

```
./models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.xml
./models/tiny_yolov3/FP32/frozen_tiny_yolov3_model.xml
./models/yolov3/FP16/frozen_yolov3_model.xml
./models/yolov3/FP32/frozen_yolov3_model.xml
```

The “non-tiny” version is more accurate, but takes more resources. And the FP16 is faster than FP32, and you lose some precision. Running using the Neural Compute Stick, I just use the tiny FP16 version, and cap my FPS on the camera streams to 10 FPS. Seems to work fine with two cameras.

# Instructions for Building

* This version (master branch of repo) is verified to be working with a fresh install of Ubuntu 18.04 LTS Desktop with a download link to OpenVINO 2019-R3.1
* Register to get a download link for the OpenVINO toolkit for Linux: https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux
* Once you receive the email, follow the link to be presented with a page that has a button called "Full Package". Right click that link and save the download link for later.
* Now follow the following instructions to use this repo's easy installer to get the build environment set up:
```bash
wget https://raw.githubusercontent.com/AndBobsYourUncle/neural_security_system/easy_installer.sh
chmod +x easy_installer.sh
sudo ./easy_installer.sh LINK_TO_OPENVINO_FULL_DOWNLOAD NON_ROOT_USER
```
* You must use the link to download the full OpenVINO package from following the link in the email you received when registering to download OpenVINO. Also, this script assumes you have some non-root user that you'd like to use to build the final executable.
* This script will take some time to complete. Once it is done, as your non-root user, run the following:
```bash
cd ~/neural_security_system
source /opt/intel/openvino/bin/setupvars.sh
make -B
```
* The build should succeed, and you should now have a working executable of this app.

# Using

You should create a YAML file that lists all cameras you'd like to monitor. Here's a sample:
```
cameras:
  - name: Front Door
    input: http://192.168.1.52:8081
    mqtt_topic: cameras/front_door/humans
    crop_top: 80
    crop_right: 150
    crop_bottom: 0
    crop_left: 0
  - name: Driveway
    input: http://192.168.1.52:8082
    mqtt_topic: cameras/driveway/humans
    crop_top: 0
    crop_right: 0
    crop_bottom: 0
    crop_left: 0
```

Running the application with the -h option yields the following usage message:

```
./neural_security_system -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

neural_security_system [OPTION]
Options:

    -h                        Print a usage message.
    -cameras "<path>"       Optional. Specify path to camera YAML config.
    -i "<path>"             Required. Path to a video file (specify "cam" to work with camera).
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"  Optional. Required for CPU custom layers.Absolute path to a shared library with the layers implementation.
          Or
      -c "<absolute_path>"  Optional. Required for GPU custom kernels.Absolute path to the .xml file with the kernels description.
    -d "<device>"           Optional. Specify a target device to infer on (CPU, GPU). The demo will look for a suitable plugin for the specified device
    -pc                       Optional. Enable per-layer performance report.
    -r                        Optional. Output inference results raw values showing.
    -t                        Optional. Probability threshold for detections.
    -iou_t                    Optional. Filtering intersection over union threshold for overlapping boxes.
    -auto_resize              Optional. Enable resizable input with support of ROI crop and auto resize.
    -mh "<mqtt_broker>"     Required. Username for the MQTT client
    -u "<mqtt_username>"    Required. Username for the MQTT client
    -p "<mqtt_password>"    Required. Password for the MQTT client
    -to "<human_timeout>"   Optional. Seconds between no people detected and MQTT publish. Default is 5
    -tp "<topic>"           Required. Topic to publish the presence of humans.
    -no_show                 Optional. Disables video out (for use as service)
    -cr "<pixels>"          Optional. Number of pixels to crop from the right.
    -cb "<pixels>"          Optional. Number of pixels to crop from the bottom.
    -cl "<pixels>"          Optional. Number of pixels to crop from the left.
    -ct "<pixels>"          Optional. Number of pixels to crop from the top.
    -async                    Optional. Start program in async mode.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To use the CPU, you _must_ pass in the path to the CPU extention when running the app. Also, before any exection, you have to run `source /opt/intel/openvino/bin/setupvars.sh`

### My Example Usage (MMJPEG security camera) w/ CPU
```bash
$ source /opt/intel/openvino/bin/setupvars.sh
$ ./neural_security_system -cameras PATH_TO_CAMERA_YAML \
-m ./models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.xml \
-d CPU -u MQTT_USER -p MQTT_PASSWORD -mh tcp://MQTT_HOST_IP:1883 \
-t 0.4 -no_show \
-l $INTEL_CVSDK_DIR/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so
```
### My Example Usage (MMJPEG security camera) w/ Intel Neural Compute Stick 2
```bash
$ source /opt/intel/openvino/bin/setupvars.sh
$ ./neural_security_system -cameras PATH_TO_CAMERA_YAML \
-m ./models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.xml \
-d MYRIAD -u MQTT_USER -p MQTT_PASSWORD -mh tcp://MQTT_HOST_IP:1883 \
-t 0.4 -no_show
```

### Output

If you run the executable without the `-no_show` command line argument, you will get a set of windows open up, each with the output from one of the cameras being monitored. You will also see any bounding boxes drawn when it detects a human.

### Building the YOLOv3 Models

The easy installer script should build all available YOLO models for you, but if you'd like to know how to do so yourself, here are the steps:

```bash
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cp ~/neural_security_system/yolo_v3_changed.json ./
cp ~/neural_security_system/yolo_v3_tiny_changed.json ./
```

Building YOLOv3 FP16 version:

```bash
python3 convert_weights_pb.py --class_names coco.names \
  --data_format NHWC --weights_file yolov3.weights \
  --output_graph frozen_yolov3_model.pb
python3 ~/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --input_model frozen_yolov3_model.pb \
  --tensorflow_use_custom_operations_config yolo_v3_changed.json \
  --input_shape [1,416,416,3] --data_type=FP16
mv frozen_yolov3_model.xml ~/neural_security_system/models/yolov3/FP16/
mv frozen_yolov3_model.bin ~/neural_security_system/models/yolov3/FP16/
cp coco.names ~/neural_security_system/models/yolov3/FP16/frozen_yolov3_model.labels
```
Building YOLOv3 FP32 version:

```bash
python3 convert_weights_pb.py --class_names coco.names \
  --data_format NHWC --weights_file yolov3.weights \
  --output_graph frozen_yolov3_model.pb
python3 ~/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --input_model frozen_yolov3_model.pb \
  --tensorflow_use_custom_operations_config yolo_v3_changed.json \
  --input_shape [1,416,416,3]
mv frozen_yolov3_model.xml ~/neural_security_system/models/yolov3/FP32/
mv frozen_yolov3_model.bin ~/neural_security_system/models/yolov3/FP32/
cp coco.names ~/neural_security_system/models/yolov3/FP32/frozen_yolov3_model.labels
```

Building Tiny YOLOv3 FP16 version:

```bash
python3 convert_weights_pb.py --class_names coco.names \
  --data_format NHWC --weights_file yolov3-tiny.weights \
  --output_graph frozen_tiny_yolov3_model.pb --tiny
python3 ~/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --input_model frozen_tiny_yolov3_model.pb \
  --tensorflow_use_custom_operations_config yolo_v3_tiny_changed.json \
  --input_shape [1,416,416,3] --data_type=FP16
mv frozen_tiny_yolov3_model.xml ~/neural_security_system/models/tiny_yolov3/FP16/
mv frozen_tiny_yolov3_model.bin ~/neural_security_system/models/tiny_yolov3/FP16/
cp coco.names ~/neural_security_system/models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.labels
```

Building Tiny YOLOv3 FP32 version:

```bash
python3 convert_weights_pb.py --class_names coco.names \
  --data_format NHWC --weights_file yolov3-tiny.weights \
  --output_graph frozen_tiny_yolov3_model.pb --tiny
python3 ~/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --input_model frozen_tiny_yolov3_model.pb \
  --tensorflow_use_custom_operations_config yolo_v3_tiny_changed.json \
  --input_shape [1,416,416,3]
mv frozen_tiny_yolov3_model.xml ~/neural_security_system/models/tiny_yolov3/FP32/
mv frozen_tiny_yolov3_model.bin ~/neural_security_system/models/tiny_yolov3/FP32/
cp coco.names ~/neural_security_system/models/tiny_yolov3/FP32/frozen_tiny_yolov3_model.labels
```
