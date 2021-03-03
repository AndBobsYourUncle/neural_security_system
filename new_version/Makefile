PAHO_C_LIB_DIR ?= /usr/local/lib
PAHO_C_INC_DIR ?= /usr/local/include

ifneq ($(CROSS_COMPILE),)
  CC  = $(CROSS_COMPILE)gcc
  CXX = $(CROSS_COMPILE)g++
  AR  = $(CROSS_COMPILE)ar
  LD  = $(CROSS_COMPILE)ld
endif

CXXFLAGS += -Wall -std=c++11
CPPFLAGS += -I.. -I$(PAHO_C_INC_DIR)
CPPFLAGS += -D_NDEBUG
CXXFLAGS += -O2
CPPFLAGS += -fPIE
CPPFLAGS += -O3

LDLIBS += -L../../lib -L$(PAHO_C_LIB_DIR) -lpaho-mqttpp3 -lpaho-mqtt3a -lyaml-cpp
LDLIBS_SSL += -L../../lib -L$(PAHO_C_LIB_DIR) -lpaho-mqttpp3 -lpaho-mqtt3as -lyaml-cpp

neural_security_system:
  $(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp -I. \
            -L./lib/ -Wl,-rpath=./lib/ \
            -I$(INTEL_CVSDK_DIR)/opencv/include/ \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include/ \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include/cpp \
            -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/lib/intel64 -linference_engine -ldl -lpthread \
            -L$(INTEL_CVSDK_DIR)/opencv/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/ \
            -I./ \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/common/format_reader/ \
            -I$(INTEL_CVSDK_DIR)/opencv/include \
            -I/usr/local/include \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/build/thirdparty/gflags/include \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include/cpp \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/extension \
            -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/bin/intel64/Release/lib \
            -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/lib/intel64 \
            -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/build/intel64/Release/lib \
            -L$(INTEL_CVSDK_DIR)/opencv/lib -ldl -linference_engine -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_imgcodecs -lgflags_nothreads \
            $< $(LDLIBS)


.PHONY: clean
clean:
    $(clean)
