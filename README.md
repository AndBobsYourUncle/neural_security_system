# Neural Security System

--

The goal of this tool is to allow monitoring of a security camera, and to actively publish the presense of humans in a boolean MQTT topic. MQTT is a pubsub type protocol. This was inspired from [https://github.com/PINTO0309/OpenVINO-YoloV3](https://github.com/PINTO0309/OpenVINO-YoloV3)

# Instructions for Building

* Download and install the OpenVINO toolkit: [https://software.intel.com/en-us/articles/OpenVINO-Install-Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux). This project assumes you have installed the toolkit as a regular user, and not root. When you get to the step to run `sudo ./install_GUI.sh`, run it without `sudo` instead.
2. Install paho MQTT library

```bash
sudo apt-get install libssl-dev

git clone https://github.com/eclipse/paho.mqtt.c.git
cd paho.mqtt.c/
git checkout v1.2.1
cmake -Bbuild -H. -DPAHO_WITH_SSL=ON -DPAHO_BUILD_SHARED=ON
sudo cmake --build build/ --target install
sudo ldconfig

git clone https://github.com/eclipse/paho.mqtt.cpp
cd paho.mqtt.cpp/
cmake -Bbuild -H. -DPAHO_WITH_SSL=ON -DPAHO_BUILD_SHARED=ON
sudo cmake --build build/ --target install
sudo ldconfig
```

* After you have installed all the prerequisites, build the sample projects. `cd ~/intel/openvino/deployment_tools/demo`, `./demo_squeezenet_download_convert_run.sh`
* Now that you have the sample built, you should be able to copy `libcpu_extension.so` to the `lib/` folder of this repo (just in case the version here is outdated). It's most likely going to be located here: `~/inference_engine_samples/intel64/Release/lib/libcpu_extension.so`
* `make -B`

# Using

Running the application with the -h option yields the following usage message:

```
./neural_security_system -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

neural_security_system [OPTION]
Options:

    -h                        Print a usage message.
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
    -no_image                 Optional. Disables video out (for use as service)
    -cr "<pixels>"          Optional. Number of pixels to crop from the right.
    -cb "<pixels>"          Optional. Number of pixels to crop from the bottom.
    -cl "<pixels>"          Optional. Number of pixels to crop from the left.
    -ct "<pixels>"          Optional. Number of pixels to crop from the top.
```

Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on GPU with a pre-trained object detection model:
### CPU + USB Camera Mode + Full size YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/YoloV3/FP32/frozen_yolo_v3.xml -d CPU
```
### MYRIAD + USB Camera Mode + Full size YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml -d MYRIAD
```
### CPU + USB Camera Mode + tiny-YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/tiny-YoloV3/FP16/frozen_yolo_v3.xml -d CPU
```
### MYRIAD + USB Camera Mode + tiny-YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml -d MYRIAD -t 0.2
```
### Movie File Mode
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/frozen_yolo_v3.xml -l ../lib/libcpu_extension.so -d CPU
```
**NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the Model Optimizer tool.

The only GUI knob is to use **Tab** to switch between the synchronized execution and the true Async mode.

### Demo Output

This program uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode, the program reports:
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and to display the results.
* **Detection time**: inference time for the object detection network. It is reported in the Sync mode only.
* **Wallclock time**, which is combined application-level performance.

### Building the YOLOv3 Models

There is an included Tony YOLO v3 model in this repo, but it may be out of date and not work with the most recent OpenVINO tollkit. Here's how to build each of the four versions of YOLOv3:

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
python3 ~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \ 
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
python3 ~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \ 
  --input_model frozen_yolov3_model.pb \ 
  --tensorflow_use_custom_operations_config yolo_v3_changed.json \
  --input_shape [1,416,416,3] --data_type=FP16
mv frozen_yolov3_model.xml ~/neural_security_system/models/yolov3/FP32/
mv frozen_yolov3_model.bin ~/neural_security_system/models/yolov3/FP32/
cp coco.names ~/neural_security_system/models/yolov3/FP32/frozen_yolov3_model.labels
```

Building Tiny YOLOv3 FP16 version:

```bash
python3 convert_weights_pb.py --class_names coco.names \ 
  --data_format NHWC --weights_file yolov3-tiny.weights \ 
  --output_graph frozen_tiny_yolov3_model.pb --tiny
python3 ~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \ 
  --input_model frozen_tiny_yolov3_model.pb \ 
  --tensorflow_use_custom_operations_config yolo_v3_tiny_changed.json \
  --input_shape [1,416,416,3] --data_type=FP16
mv frozen_tiny_yolov3_model.xml ~/neural_security_system/models/tiny_yolov3/FP16/
mv frozen_tiny_yolov3_model.bin ~/neural_security_system/models/tiny_yolov3/FP16/
cp coco.names ~/neural_security_system/models/tiny_yolov3/FP16/frozen_yolov3_model.labels
```

Building Tiny YOLOv3 FP32 version:

```bash
python3 convert_weights_pb.py --class_names coco.names \ 
  --data_format NHWC --weights_file yolov3-tiny.weights \ 
  --output_graph frozen_tiny_yolov3_model.pb --tiny
python3 ~/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \ 
  --input_model frozen_tiny_yolov3_model.pb \ 
  --tensorflow_use_custom_operations_config yolo_v3_tiny_changed.json \
  --input_shape [1,416,416,3]
mv frozen_tiny_yolov3_model.xml ~/neural_security_system/models/tiny_yolov3/FP32/
mv frozen_tiny_yolov3_model.bin ~/neural_security_system/models/tiny_yolov3/FP32/
cp coco.names ~/neural_security_system/models/tiny_yolov3/FP32/frozen_yolov3_model.labels
```
