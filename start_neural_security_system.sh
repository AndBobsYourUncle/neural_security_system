#!/bin/bash

source $INSTALL_DIR/bin/setupvars.sh

./neural_security_system -cameras $CAMERAS -m $MODEL -d $DEVICE -u $MQTT_USER -p $MQTT_PASSWORD -mh $MQTT_HOST -no_show -l $INTEL_CVSDK_DIR/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so
