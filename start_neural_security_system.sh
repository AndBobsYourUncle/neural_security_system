#!/bin/bash

signalListener() {
    "$@" &
    pid="$!"
    trap "echo 'Stopping PID $pid'; kill -SIGTERM $pid" SIGINT SIGTERM

    # A signal emitted while waiting will make the wait command return code > 128
    # Let's wrap it in a loop that doesn't end before the process is indeed stopped
    while kill -0 $pid > /dev/null 2>&1; do
        wait
    done
}

source $INSTALL_DIR/bin/setupvars.sh

signalListener ./neural_security_system -cameras $CAMERAS -m $MODEL -d $DEVICE -u $MQTT_USER -p $MQTT_PASSWORD -mh $MQTT_HOST -no_show -l $INTEL_CVSDK_DIR/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so
