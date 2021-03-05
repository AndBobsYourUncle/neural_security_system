FROM ubuntu:18.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    sed -i 's/=DEFAULTS/=intel-openvino-ie-sdk-ubuntu-bionic__x86_64;intel-openvino-ie-rt-cpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-vpu-ubuntu-bionic__x86_64;intel-openvino-model-optimizer__x86_64;intel-openvino-opencv-lib-ubuntu-bionic__x86_64/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/cpp/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/cpp/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"

WORKDIR $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites
RUN ./install_prerequisites.sh

RUN apt-get update && apt-get install -y --no-install-recommends dh-autoreconf unzip && \
    cd /tmp/ && \
    wget https://github.com/libusb/libusb/archive/v1.0.22.zip && \
    unzip v1.0.22.zip && cd libusb-1.0.22 && \
    ./bootstrap.sh && \
    ./configure --disable-udev --enable-shared && \
    make -j4 && make install && \
    rm -rf /tmp/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libssl-dev \
    libgflags-dev \
    build-essential \
    gcc \
    make \
    cmake \
    cmake-gui \
    cmake-curses-gui && \
    rm -rf /var/lib/apt/lists/*

RUN cd /usr && git clone https://github.com/eclipse/paho.mqtt.c.git && \
    cd /usr/paho.mqtt.c && git checkout v1.3.1 && \
    cmake -Bbuild -H. -DPAHO_WITH_SSL=ON -DPAHO_BUILD_SHARED=ON && \
    cmake --build build/ --target install && \
    ldconfig && \
    cd /usr && \
    rm -rf /usr/paho.mqtt.c

RUN cd /usr && git clone https://github.com/eclipse/paho.mqtt.cpp && \
    cd /usr/paho.mqtt.cpp && git checkout v1.1 && \
    cmake -Bbuild -H. -DPAHO_WITH_SSL=ON -DPAHO_BUILD_SHARED=ON && \
    cmake --build build/ --target install && \
    ldconfig && \
    cd /usr && \
    rm -rf /usr/paho.mqtt.cpp

RUN cd /usr && git clone https://github.com/jbeder/yaml-cpp.git && \
    cd /usr/yaml-cpp/ && git checkout yaml-cpp-0.6.3 && mkdir build && \
    cd /usr/yaml-cpp/build && \
    cmake -DYAML_BUILD_SHARED_LIBS=ON .. && \
    cmake --build ./ --target install && \
    ldconfig && \
    cd /usr && \
    rm -rf /usr/yaml-cpp

RUN pip3 install image && \
    cd /usr && git clone https://github.com/mystic123/tensorflow-yolo-v3.git && \
    cd tensorflow-yolo-v3 && \
    git checkout ed60b90 && \
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names && \
    wget https://pjreddie.com/media/files/yolov3.weights && \
    wget https://pjreddie.com/media/files/yolov3-tiny.weights && \
    cp $INSTALL_DIR/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json ./ && \
    cp $INSTALL_DIR/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json ./ && \

    mkdir -p /usr/neural_security_system/models/tiny_yolov3/FP16 && \
    cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3-tiny.weights \
      --output_graph frozen_tiny_yolov3_model.pb --tiny && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_tiny_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3_tiny.json \
      --input_shape [1,416,416,3] --data_type=FP16 && \
    mv frozen_tiny_yolov3_model.xml /usr/neural_security_system/models/tiny_yolov3/FP16/ && \
    mv frozen_tiny_yolov3_model.bin /usr/neural_security_system/models/tiny_yolov3/FP16/ && \
    cp coco.names /usr/neural_security_system/models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.labels && \

    mkdir -p /usr/neural_security_system/models/tiny_yolov3/FP32 && \
    cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3-tiny.weights \
      --output_graph frozen_tiny_yolov3_model.pb --tiny && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_tiny_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3_tiny.json \
      --input_shape [1,416,416,3] && \
    mv frozen_tiny_yolov3_model.xml /usr/neural_security_system/models/tiny_yolov3/FP32/ && \
    mv frozen_tiny_yolov3_model.bin /usr/neural_security_system/models/tiny_yolov3/FP32/ && \
    cp coco.names /usr/neural_security_system/models/tiny_yolov3/FP32/frozen_tiny_yolov3_model.labels && \

    mkdir -p /usr/neural_security_system/models/yolov3/FP16 && \
    cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3.weights \
      --output_graph frozen_yolov3_model.pb && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3.json \
      --input_shape [1,416,416,3] --data_type=FP16 && \
    mv frozen_yolov3_model.xml /usr/neural_security_system/models/yolov3/FP16/ && \
    mv frozen_yolov3_model.bin /usr/neural_security_system/models/yolov3/FP16/ && \
    cp coco.names /usr/neural_security_system/models/yolov3/FP16/frozen_yolov3_model.labels && \

    mkdir -p /usr/neural_security_system/models/yolov3/FP32 && \
    cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3.weights \
      --output_graph frozen_yolov3_model.pb && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3.json \
      --input_shape [1,416,416,3] && \
    mv frozen_yolov3_model.xml /usr/neural_security_system/models/yolov3/FP32/ && \
    mv frozen_yolov3_model.bin /usr/neural_security_system/models/yolov3/FP32/ && \
    cp coco.names /usr/neural_security_system/models/yolov3/FP32/frozen_yolov3_model.labels && \

    cd /usr && \
    rm -rf /usr/tensorflow-yolo-v3

ENV INSTALL_DIR=$INSTALL_DIR

ENV THRESHOLD 0.5
ENV IOU_THRESHOLD 0.4
ENV HUMAN_TIMEOUT 5

WORKDIR /usr/neural_security_system

RUN touch /usr/neural_security_system/cameras.yaml

COPY . /usr/neural_security_system

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && make -B"

CMD [ "/bin/bash", "-c", "/usr/neural_security_system/start_neural_security_system.sh" ]
