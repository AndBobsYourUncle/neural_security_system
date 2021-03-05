/*
// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the neural securty system application. Created based on the
* OpenVINO Open Model Zoo object_detection_demo_ssd_async application.
*/

#include <iostream>
#include <vector>
#include <string>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/args_helper.hpp>
#include <samples/slog.hpp>
#include <samples/images_capture.h>
#include <samples/default_flags.hpp>
#include <unordered_map>
#include <gflags/gflags.h>

#include <iostream>

#include <samples/performance_metrics.hpp>

#include "pipelines/async_pipeline.h"
#include "pipelines/config_factory.h"
#include "pipelines/metadata.h"
#include "models/detection_model_yolo.h"
#include "models/detection_model_ssd.h"

#include <csignal>
#include "mqtt/async_client.h"
#include "yaml-cpp/yaml.h"

const int QOS = 1;

const auto PERIOD = std::chrono::seconds(5);

const int MAX_BUFFERED_MSGS = 120;  // 120 * 5sec => 10min off-line buffering

const std::string PERSIST_DIR { "data-persist" };

static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Architecture type: ssd or yolo";
static const char video_message[] = "Required. Path to a video file (specify \"cam\" to work with camera).";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
"The application will look for a suitable plugin for a specified device.";
static const char labels_message[] = "Optional. Path to a file with labels mapping.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
"Absolute path to the .xml file with the kernel descriptions.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
"Absolute path to a shared library with the kernel implementations.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char raw_output_message[] = "Optional. Inference results as raw values.";
static const char input_resizable_message[] = "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char num_inf_req_message[] = "Optional. Number of infer requests.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char iou_thresh_output_message[] = "Optional. Filtering intersection over union threshold for overlapping boxes (YOLOv3 only).";
static const char yolo_af_message[] = "Optional. Use advanced postprocessing/filtering algorithm for YOLO.";

static const char host_message[] = "Required. Host for MQTT server.";
static const char user_message[] = "Required. User for MQTT server.";
static const char pass_message[] = "Required. Password for MQTT server.";

static const char alive_message[] = "Required. MQTT topic for alive sigal.";
static const char will_message[] = "Optional. MQTT topic for LWT sigal.";

static const char detection_window_message[] = "Optional. Minimum human frames to trigger detection. Default is 5.";
static const char timeout_message[] = "Optional. Seconds between no people detected and MQTT publish. Default is 5.";

static const char mqtt_topic_message[] = "Required. Specify an MQTT topic.";

static const char crop_right_message[] = "Optional. Number of pixels to crop from the right.";
static const char crop_bottom_message[] = "Optional. Number of pixels to crop from the bottom.";
static const char crop_left_message[] = "Optional. Number of pixels to crop from the left.";
static const char crop_top_message[] = "Optional. Number of pixels to crop from the top.";

static const char cameras_message[] = "Required. Specify path to camera YAML config.";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(i, "", video_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(labels, "", labels_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.7, thresh_output_message);
DEFINE_double(iou_t, 0.4, iou_thresh_output_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_uint32(nireq, 2, num_inf_req_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(loop, false, loop_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_bool(yolo_af, false, yolo_af_message);

DEFINE_string(host, "", host_message);
DEFINE_string(user, "", user_message);
DEFINE_string(pass, "", pass_message);

DEFINE_string(alive, "", alive_message);
DEFINE_string(will, "", will_message);

DEFINE_double(dw, 5, detection_window_message);

DEFINE_double(to, 5, timeout_message);

DEFINE_string(tp, "", mqtt_topic_message);

DEFINE_double(cr, 0, crop_right_message);
DEFINE_double(cb, 0, crop_bottom_message);
DEFINE_double(cl, 0, crop_left_message);
DEFINE_double(ct, 0, crop_top_message);

DEFINE_string(cameras, "", cameras_message);

struct CustomImageMetaData : public MetaData {
    cv::Mat img;
    std::chrono::steady_clock::time_point timeStamp;
    uint cameraIndex;

    CustomImageMetaData() {
    }

    CustomImageMetaData(cv::Mat img, std::chrono::steady_clock::time_point timeStamp, uint cameraIndex):
        img(img),
        timeStamp(timeStamp),
        cameraIndex(cameraIndex) {
    }
};

bool exitGracefully = false;

void signalHandler(int signum) {
   std::cout << "Interrupt signal (" << signum << ") received. Exiting gracefully...\n";

   exitGracefully = true;
}

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "neural_security_system [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i \"<path>\"               " << video_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -labels \"<path>\"          " << labels_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -r                        " << raw_output_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << num_inf_req_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_processed_video << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -yolo_af                  " << yolo_af_message << std::endl;
    std::cout << "    -host                     " << host_message << std::endl;
    std::cout << "    -user                     " << user_message << std::endl;
    std::cout << "    -pass                     " << pass_message << std::endl;
    std::cout << "    -alive                    " << alive_message << std::endl;
    std::cout << "    -will                     " << alive_message << std::endl;
    std::cout << "    -dw                       " << detection_window_message << std::endl;
    std::cout << "    -to                       " << timeout_message << std::endl;
    std::cout << "    -cr                       " << crop_right_message << std::endl;
    std::cout << "    -cb                       " << crop_bottom_message << std::endl;
    std::cout << "    -cl                       " << crop_left_message << std::endl;
    std::cout << "    -ct                       " << crop_top_message << std::endl;
    std::cout << "    -cameras                  " << cameras_message << std::endl;
}


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_host.empty()) {
        throw std::logic_error("Parameter -host is not set");
    }

    if (FLAGS_user.empty()) {
        throw std::logic_error("Parameter -user is not set");
    }

    if (FLAGS_pass.empty()) {
        throw std::logic_error("Parameter -pass is not set");
    }

    if (FLAGS_alive.empty()) {
        throw std::logic_error("Parameter -alive is not set");
    }

    if (FLAGS_i.empty() && FLAGS_cameras.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    return true;
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(const DetectionResult& result) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<CustomImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }

    // Visualizing result data over source image
    if (FLAGS_r) {
        slog::info << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;
    }

    for (auto obj : result.objects) {
        if (FLAGS_r) {
            slog::info << " "
                       << std::left << std::setw(9) << obj.label << " | "
                       << std::setw(10) << obj.confidence << " | "
                       << std::setw(4) << std::max(int(obj.x), 0) << " | "
                       << std::setw(4) << std::max(int(obj.y), 0) << " | "
                       << std::setw(4) << std::min(int(obj.width), outputImg.cols) << " | "
                       << std::setw(4) << std::min(int(obj.height), outputImg.rows)
                       << slog::endl;
        }

        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(3) << obj.confidence;

        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
            cv::Scalar(0, 0, 255));
        cv::rectangle(outputImg, obj, cv::Scalar(0, 0, 255));
    }

    return outputImg;
}


int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try {
        slog::info << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        uint numCameras;

        YAML::Node config;

        if(FLAGS_cameras != "") {
            config = YAML::LoadFile(FLAGS_cameras);

            numCameras = config["cameras"].size();
        } else {
            numCameras = 1;
        }

        std::string cameraNames[numCameras];
        std::string cameraInputs[numCameras];
        std::string cameraTopics[numCameras];
        int camerasCT[numCameras];
        int camerasCR[numCameras];
        int camerasCB[numCameras];
        int camerasCL[numCameras];

        if(FLAGS_cameras != "") {
            slog::info << "Initialize camera config from yaml: " << FLAGS_cameras << slog::endl << slog::endl;

            const YAML::Node& cameras = config["cameras"];

            for (std::size_t i=0;i<numCameras;i++) {
                const YAML::Node camera = cameras[i];

                cameraNames[i] = camera["name"].as<std::string>();
                cameraInputs[i] = camera["input"].as<std::string>();
                cameraTopics[i] = camera["mqtt_topic"].as<std::string>();
                camerasCT[i] = camera["crop_top"].as<int>();
                camerasCR[i] = camera["crop_right"].as<int>();
                camerasCB[i] = camera["crop_bottom"].as<int>();
                camerasCL[i] = camera["crop_left"].as<int>();

                slog::info << "Camera " << i << ": " << slog::endl;
                std::cout << "name: " << camera["name"].as<std::string>() << std::endl;
                std::cout << "input: " << camera["input"].as<std::string>() << std::endl;
                std::cout << "mqtt_topic: " << camera["mqtt_topic"].as<std::string>() << std::endl << std::endl;
            }
        } else {
            std::cout << "Initialize camera config from arguments: " << std::endl << std::endl;

            cameraNames[0] = "Camera";
            cameraInputs[0] = FLAGS_i;
            cameraTopics[0] = FLAGS_tp;
            camerasCT[0] = FLAGS_ct;
            camerasCR[0] = FLAGS_cr;
            camerasCB[0] = FLAGS_cb;
            camerasCL[0] = FLAGS_cl;

            std::cout << "name: " << cameraNames[0] << std::endl;
            std::cout << "input: " << cameraInputs[0] << std::endl;
            std::cout << "mqtt_topic: " << cameraTopics[0] << std::endl << std::endl;
        }

        PerformanceMetrics cameraMetrics[numCameras];

        std::string address = FLAGS_host;

        mqtt::async_client cli(address, "");

        mqtt::connect_options connOpts;

        if(FLAGS_will != "") {
            mqtt::message willmsg(FLAGS_will, "unexpected exit", 1, true);
            mqtt::will_options will(willmsg);
            connOpts.set_will(will);
        }

        connOpts.set_keep_alive_interval(MAX_BUFFERED_MSGS * PERIOD);
        connOpts.set_clean_session(true);
        connOpts.set_automatic_reconnect(true);
        connOpts.set_user_name(FLAGS_user);
        connOpts.set_password(FLAGS_pass);

        // Connect to the MQTT broker
        std::cout << "Connecting to server '" << address << "'..." << std::flush;
        cli.connect(connOpts)->wait();
        std::cout << "OK\n" << std::endl;

        // Initial publish for "off"
        mqtt::topic::ptr_t aliveTopic;
        aliveTopic = mqtt::topic::create(cli, FLAGS_alive, QOS, true);
        mqtt::topic current_topic(*aliveTopic);
        current_topic.publish(std::move("OFF"));

        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;

        std::unique_ptr<ImagesCapture> caps[numCameras];

        for (std::size_t i=0;i<numCameras;i++) {
            caps[i] = openImagesCapture(cameraInputs[i], FLAGS_loop);
        }

        cv::Mat curr_frame;

        //------------------------------ Running Detection routines ----------------------------------------------
        std::vector<std::string> labels;
        if (!FLAGS_labels.empty())
            labels = DetectionModel::loadLabels(FLAGS_labels);

        std::unique_ptr<ModelBase> model;
        if (FLAGS_at == "ssd") {
            model.reset(new ModelSSD(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, labels));
        }
        else if (FLAGS_at == "yolo") {
            model.reset(new ModelYolo3(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, FLAGS_yolo_af, (float)FLAGS_iou_t, labels));
        }
        else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        InferenceEngine::Core core;
        AsyncPipeline pipeline(std::move(model),
            ConfigFactory::getUserConfig(FLAGS_d, FLAGS_l, FLAGS_c, FLAGS_pc, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
            core);
        Presenter presenter;

        int64_t frameNum = -1;
        std::unique_ptr<ResultBase> result;

        bool hasPeopleInFrame = false;

        uint cameraIndex = 0;

        uint humanDetectedFrames[numCameras];
        std::chrono::time_point<std::chrono::high_resolution_clock> timeHumansDetected[numCameras];
        bool humansDetected[numCameras];
        bool humansNotified[numCameras];

        // Create a topic object. This is a conventience since we will
        // repeatedly publish messages with the same parameters.
        mqtt::topic::ptr_t topics[numCameras];
        for (std::size_t i=0;i<numCameras;i++) {
            topics[i] = mqtt::topic::create(cli, cameraTopics[i], QOS, true);

            // Initial publish for "off"
            mqtt::topic currentTopic(*topics[i]);
            currentTopic.publish(std::move("OFF"));
        }

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;

        bool aliveSignalSent = false;
        bool allCamerasStarted = false;

        while (!exitGracefully) {
            if (pipeline.isReadyToProcess()) {
                //--- Capturing frame. If previous frame hasn't been inferred yet, reuse it instead of capturing new one
                auto startTime = std::chrono::steady_clock::now();
                curr_frame = caps[cameraIndex]->read();
                if (curr_frame.empty()) {
                    if (frameNum == -1) {
                        throw std::logic_error("Can't read an image from the input");
                    }
                    else {
                        // Input stream is over
                        break;
                    }
                }

                cv::Size imageSize = curr_frame.size();

                double cropWidth = imageSize.width - camerasCR[cameraIndex] - camerasCL[cameraIndex];
                double cropHeight = imageSize.height - camerasCT[cameraIndex] - camerasCB[cameraIndex];

                // Setup a rectangle to define your region of interest
                cv::Rect myROI(camerasCL[cameraIndex], camerasCT[cameraIndex], cropWidth, cropHeight);

                // Crop the full image to that image contained by the rectangle myROI
                // Note that this doesn't copy the data
                cv::Mat croppedFrame(curr_frame, myROI);

                frameNum = pipeline.submitData(ImageInputData(croppedFrame),
                    std::make_shared<CustomImageMetaData>(croppedFrame, startTime, cameraIndex));
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to DetectionResult*
            //    and use your own processing instead of calling renderDetectionData().
            while ((result = pipeline.getResult()) && !exitGracefully) {
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>());

                hasPeopleInFrame = false;
                for (auto &object : result->asRef<DetectionResult>().objects) {
                    if (object.confidence < FLAGS_t)
                        continue;

                    if(object.label == "person")
                        hasPeopleInFrame = true;
                }

                uint resultCameraIndex = result->metaData->asRef<CustomImageMetaData>().cameraIndex;

                //--- Showing results and device information
                presenter.drawGraphs(outFrame);

                cameraMetrics[resultCameraIndex].update(result->metaData->asRef<CustomImageMetaData>().timeStamp,
                    outFrame, { 10,22 }, 0.65);

                // metrics.update(result->metaData->asRef<CustomImageMetaData>().timeStamp,
                //     outFrame, { 10,22 }, 0.65);
                if (!FLAGS_no_show) {
                    cv::imshow(cameraNames[resultCameraIndex], outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        exitGracefully = true;
                    }
                    else {
                        presenter.handleKey(key);
                    }
                }

                mqtt::topic currentTopic(*topics[resultCameraIndex]);

                if(hasPeopleInFrame && !humansDetected[resultCameraIndex]) {
                    humansDetected[resultCameraIndex] = true;
                    humanDetectedFrames[resultCameraIndex] = 0;
                }

                if(hasPeopleInFrame) {
                    timeHumansDetected[resultCameraIndex] = std::chrono::high_resolution_clock::now();

                    if(!humansNotified[resultCameraIndex]) {
                        humanDetectedFrames[resultCameraIndex]++;

                        if(humanDetectedFrames[resultCameraIndex] >= FLAGS_dw) {
                            humansNotified[resultCameraIndex] = true;
                            currentTopic.publish(std::move("ON"));
                        }
                    }
                }

                if(!hasPeopleInFrame && humansDetected[resultCameraIndex]) {
                    auto timeNoHumans = std::chrono::high_resolution_clock::now();

                     std::chrono::milliseconds timeSinceHumans = std::chrono::duration_cast<std::chrono::milliseconds>(timeNoHumans - timeHumansDetected[resultCameraIndex]);

                    if(timeSinceHumans.count() > (FLAGS_to * 1000)) {
                        humansDetected[resultCameraIndex] = false;
                        humansNotified[resultCameraIndex] = false;
                        currentTopic.publish(std::move("OFF"));
                    }
                }
            }

            cameraIndex++;

            if(cameraIndex >= numCameras) {
                allCamerasStarted = true;
                cameraIndex = 0;
            }

            if(allCamerasStarted && !aliveSignalSent) {
                mqtt::topic current_topic(*aliveTopic);
                current_topic.publish(std::move("ON"));

                aliveSignalSent = true;
            }
        }

        //// ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        while (result = pipeline.getResult()) {
            cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>());

            uint resultCameraIndex = result->metaData->asRef<CustomImageMetaData>().cameraIndex;

            //--- Showing results and device information
            presenter.drawGraphs(outFrame);
            cameraMetrics[resultCameraIndex].update(result->metaData->asRef<CustomImageMetaData>().timeStamp,
                outFrame, { 10, 22 }, 0.65);
            if (!FLAGS_no_show) {
                cv::imshow(cameraNames[resultCameraIndex], outFrame);
                //--- Updating output window
                cv::waitKey(1);
            }
        }

        //// --------------------------- Report metrics -------------------------------------------------------
        slog::info << slog::endl << "Metric reports:" << slog::endl;
        for (std::size_t i=0;i<numCameras;i++) {
            std::cout << cameraNames[i] << ":" << std::endl;

            cameraMetrics[i].printTotal();

            std::cout << std::endl;
        }

        slog::info << presenter.reportMeans() << slog::endl;

        mqtt::topic notAliveTopic(*aliveTopic);
        notAliveTopic.publish(std::move("OFF"));
    }
    catch (const std::exception& error) {
        slog::err << "[ ERROR ] " << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "[ ERROR ] Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << slog::endl << "The execution has completed successfully" << slog::endl;
    return 0;
}
