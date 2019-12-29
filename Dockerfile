FROM ubuntu:18.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
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
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"

WORKDIR $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites
RUN ./install_prerequisites.sh

WORKDIR $INSTALL_DIR/deployment_tools/demo

RUN ./demo_squeezenet_download_convert_run.sh

RUN ./demo_security_barrier_camera.sh || :

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
    ldconfig

RUN cd /usr && git clone https://github.com/eclipse/paho.mqtt.cpp && \
    cd /usr/paho.mqtt.cpp && git checkout v1.1 && \
    cmake -Bbuild -H. -DPAHO_WITH_SSL=ON -DPAHO_BUILD_SHARED=ON && \
    cmake --build build/ --target install && \
    ldconfig

RUN cd /usr && git clone https://github.com/jbeder/yaml-cpp.git && \
    cd /usr/yaml-cpp/ && git checkout yaml-cpp-0.6.3 && mkdir build && \
    cd /usr/yaml-cpp/build && \
    cmake -DYAML_BUILD_SHARED_LIBS=ON .. && \
    cmake --build ./ --target install && \
    ldconfig

RUN pip3 install image

RUN cd /usr && git clone https://github.com/mystic123/tensorflow-yolo-v3.git && \
    cd tensorflow-yolo-v3 && \
    git checkout ed60b90 && \
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names && \
    wget https://pjreddie.com/media/files/yolov3.weights && \
    wget https://pjreddie.com/media/files/yolov3-tiny.weights && \
    cp $INSTALL_DIR/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json ./ && \
    cp $INSTALL_DIR/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json ./

RUN mkdir /usr/neural_security_system && \
    mkdir /usr/neural_security_system/models && \
    mkdir /usr/neural_security_system/models/yolov3 && \
    mkdir /usr/neural_security_system/models/yolov3/FP16

RUN cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3.weights \
      --output_graph frozen_yolov3_model.pb && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3.json \
      --input_shape [1,416,416,3] --data_type=FP16 && \
    mv frozen_yolov3_model.xml /usr/neural_security_system/models/yolov3/FP16/ && \
    mv frozen_yolov3_model.bin /usr/neural_security_system/models/yolov3/FP16/ && \
    cp coco.names /usr/neural_security_system/models/yolov3/FP16/frozen_yolov3_model.labels

RUN mkdir /usr/neural_security_system/models/yolov3/FP32

RUN cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3.weights \
      --output_graph frozen_yolov3_model.pb && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3.json \
      --input_shape [1,416,416,3] && \
    mv frozen_yolov3_model.xml /usr/neural_security_system/models/yolov3/FP32/ && \
    mv frozen_yolov3_model.bin /usr/neural_security_system/models/yolov3/FP32/ && \
    cp coco.names /usr/neural_security_system/models/yolov3/FP32/frozen_yolov3_model.labels

RUN mkdir /usr/neural_security_system/models/tiny_yolov3 && \
    mkdir /usr/neural_security_system/models/tiny_yolov3/FP16

RUN cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3-tiny.weights \
      --output_graph frozen_tiny_yolov3_model.pb --tiny && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_tiny_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3_tiny.json \
      --input_shape [1,416,416,3] --data_type=FP16 && \
    mv frozen_tiny_yolov3_model.xml /usr/neural_security_system/models/tiny_yolov3/FP16/ && \
    mv frozen_tiny_yolov3_model.bin /usr/neural_security_system/models/tiny_yolov3/FP16/ && \
    cp coco.names /usr/neural_security_system/models/tiny_yolov3/FP16/frozen_tiny_yolov3_model.labels

RUN mkdir /usr/neural_security_system/models/tiny_yolov3/FP32

RUN cd /usr/tensorflow-yolo-v3 && \
    python3 convert_weights_pb.py --class_names coco.names \
      --data_format NHWC --weights_file yolov3-tiny.weights \
      --output_graph frozen_tiny_yolov3_model.pb --tiny && \
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
      --input_model frozen_tiny_yolov3_model.pb \
      --tensorflow_use_custom_operations_config yolo_v3_tiny.json \
      --input_shape [1,416,416,3] && \
    mv frozen_tiny_yolov3_model.xml /usr/neural_security_system/models/tiny_yolov3/FP32/ && \
    mv frozen_tiny_yolov3_model.bin /usr/neural_security_system/models/tiny_yolov3/FP32/ && \
    cp coco.names /usr/neural_security_system/models/tiny_yolov3/FP32/frozen_tiny_yolov3_model.labels

ENV INSTALL_DIR=$INSTALL_DIR

WORKDIR /usr/neural_security_system

COPY . /usr/neural_security_system

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && make -B"

CMD [ "/usr/neural_security_system/start_neural_security_system.sh" ]
