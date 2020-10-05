// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov3_async/main.cpp
* \example object_detection_demo_yolov3_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <unistd.h>

#include <inference_engine.hpp>

#include <ocv_common.hpp>
#include <slog.hpp>

#include <csignal>
using namespace std::chrono;
using namespace std;

#include "main.hpp"

#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

#include "mqtt/async_client.h"

#include "yaml-cpp/yaml.h"

const int    QOS = 1;

const auto PERIOD = seconds(5);

const int MAX_BUFFERED_MSGS = 120;  // 120 * 5sec => 10min off-line buffering

const string PERSIST_DIR { "data-persist" };

bool exit_gracefully = false;

void signalHandler(int signum) {
   std::cout << "Interrupt signal (" << signum << ") received. Exiting gracefully...\n";

   exit_gracefully = true;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if(FLAGS_cameras.empty()) {
        if (FLAGS_i.empty()) {
            throw std::logic_error("Parameter -i is not set");
        }

        if (FLAGS_tp.empty()) {
            throw std::logic_error("Parameter -tp is not set");
        }
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_mh.empty()) {
        throw std::logic_error("Parameter -mh is not set");
    }

    if (FLAGS_u.empty()) {
        throw std::logic_error("Parameter -u is not set");
    }

    if (FLAGS_p.empty()) {
        throw std::logic_error("Parameter -p is not set");
    }

    return true;
}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------
    auto num = layer->GetParamAsInt("num");
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    try {
        auto mask = layer->GetParamAsInts("mask");
        num = mask.size();

        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    } catch (...) {}

    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try {
        uint num_cameras;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        YAML::Node config;

        if(FLAGS_cameras != "") {
            config = YAML::LoadFile(FLAGS_cameras);

            num_cameras = config["cameras"].size();
        } else {
            num_cameras = 1;
        }

        std::string camera_names[num_cameras];
        std::string camera_inputs[num_cameras];
        std::string camera_topics[num_cameras];
        int cameras_ct[num_cameras];
        int cameras_cr[num_cameras];
        int cameras_cb[num_cameras];
        int cameras_cl[num_cameras];

        if(FLAGS_cameras != "") {
            std::cout << "Initialize camera config from yaml: " << FLAGS_cameras << "\n\n";

            const YAML::Node& cameras = config["cameras"];

            for (std::size_t i=0;i<num_cameras;i++) {
                const YAML::Node camera = cameras[i];

                camera_names[i] = camera["name"].as<std::string>();
                camera_inputs[i] = camera["input"].as<std::string>();
                camera_topics[i] = camera["mqtt_topic"].as<std::string>();
                cameras_ct[i] = camera["crop_top"].as<int>();
                cameras_cr[i] = camera["crop_right"].as<int>();
                cameras_cb[i] = camera["crop_bottom"].as<int>();
                cameras_cl[i] = camera["crop_left"].as<int>();

                std::cout << "name: " << camera["name"].as<std::string>() << "\n";
                std::cout << "input: " << camera["input"].as<std::string>() << "\n";
                std::cout << "mqtt_topic: " << camera["mqtt_topic"].as<std::string>() << "\n\n";
            }
        } else {
            std::cout << "Initialize camera config from arguments: " << "\n\n";

            camera_names[0] = "Camera";
            camera_inputs[0] = FLAGS_i;
            camera_topics[0] = FLAGS_tp;
            cameras_ct[0] = FLAGS_ct;
            cameras_cr[0] = FLAGS_cr;
            cameras_cb[0] = FLAGS_cb;
            cameras_cl[0] = FLAGS_cl;

            std::cout << "name: " << camera_names[0] << "\n";
            std::cout << "input: " << camera_inputs[0] << "\n";
            std::cout << "mqtt_topic: " << camera_topics[0] << "\n\n";
        }

        string address = FLAGS_mh;

        mqtt::async_client cli(address, "", MAX_BUFFERED_MSGS, PERSIST_DIR);

        mqtt::connect_options connOpts;

        mqtt::connect_options conopts;
        mqtt::message willmsg("cameras/lwt", "unexpected exit", 1, true);
        mqtt::will_options will(willmsg);
        connOpts.set_will(will);

        connOpts.set_keep_alive_interval(MAX_BUFFERED_MSGS * PERIOD);
        connOpts.set_clean_session(true);
        connOpts.set_automatic_reconnect(true);
        connOpts.set_user_name(FLAGS_u);
        connOpts.set_password(FLAGS_p);

        // Connect to the MQTT broker
        cout << "Connecting to server '" << address << "'..." << flush;
        cli.connect(connOpts)->wait();
        cout << "OK\n" << endl;

        // Initial publish for "off"
        mqtt::topic::ptr_t alive_topic;
        alive_topic = mqtt::topic::create(cli, "cameras/connection_alive", QOS, true);
        mqtt::topic current_topic(*alive_topic);
        current_topic.publish(std::move("OFF"));

        /** This demo covers a certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /**Loading extensions to the devices **/

#ifdef WITH_EXTENSIONS
        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from the "extension" folder containing
             * custom CPU layer implementations.
            **/
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        }
#endif

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        /** Reading network model **/
        auto cnnNetwork = ie.ReadNetwork(FLAGS_m);
        /** Setting batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        cnnNetwork.setBatchSize(1);
        /** Reading labels (if specified) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** YOLOV3-based network should have one input and three output **/
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }
        // --------------------------------- Preparing output blobs -------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, FLAGS_d);

        // -----------------------------------------------------------------------------------------------------

        // -----------------------------------------------------------------------------------------------------
        InferRequest::Ptr async_infer_request_curr[num_cameras];
        for (uint i=0 ; i < num_cameras; i++) {
            async_infer_request_curr[i] = network.CreateInferRequestPtr();
        }

        uint camera_index = 0;

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;

        cv::VideoCapture caps[num_cameras];;
        cv::Mat frames[num_cameras];
        size_t widths[num_cameras];
        size_t heights[num_cameras];

        for (std::size_t i=0;i<num_cameras;i++) {
            cout << "Camera index: " << i << endl;

            if (!((camera_inputs[i] == "cam") ? caps[i].open(0) : caps[i].open(camera_inputs[i].c_str()))) {
                throw std::logic_error("Cannot open input file or camera: " + camera_inputs[i]);
            }

            // caps[i].set(cv::CAP_PROP_BUFFERSIZE, 3);

            // read input (video) frame
            caps[i] >> frames[i];

            widths[i]  = (size_t) caps[i].get(cv::CAP_PROP_FRAME_WIDTH);
            heights[i] = (size_t) caps[i].get(cv::CAP_PROP_FRAME_HEIGHT);

            if (!caps[i].grab()) {
                throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                       "Failed to get next frame from the " + camera_inputs[i]);
            }
        }

        // --------------------------- 5. Creating infer request -----------------------------------------------
        // InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        bool isLastFrame = false;

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        std::chrono::time_point<std::chrono::high_resolution_clock> wallclock[num_cameras];
        std::chrono::time_point<std::chrono::high_resolution_clock> t0[num_cameras];
        std::chrono::time_point<std::chrono::high_resolution_clock> t1[num_cameras];
        double ocv_decode_time[num_cameras];
        double ocv_render_time[num_cameras];

        std::chrono::time_point<std::chrono::high_resolution_clock> time_humans_detected[num_cameras];
        bool humans_detected[num_cameras];

        for (std::size_t i=0;i<num_cameras;i++) {
            wallclock[i] = std::chrono::high_resolution_clock::now();
            t0[i] = std::chrono::high_resolution_clock::now();
            t1[i] = std::chrono::high_resolution_clock::now();
            ocv_decode_time[i] = 0;
            ocv_render_time[i] = 0;

            time_humans_detected[i] = std::chrono::high_resolution_clock::now();
            humans_detected[i] = false;
        }

        // Create a topic object. This is a conventience since we will
        // repeatedly publish messages with the same parameters.
        mqtt::topic::ptr_t topics[num_cameras];
        for (std::size_t i=0;i<num_cameras;i++) {
            topics[i] = mqtt::topic::create(cli, camera_topics[i], QOS, true);

            // Initial publish for "off"
            mqtt::topic current_topic(*topics[i]);
            current_topic.publish(std::move("OFF"));
        }

        bool all_cameras_started = false;

        bool alive_signal_sent = false;

        while (!exit_gracefully) {
            cv::Mat source_frame;

            t0[camera_index] = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the Async mode, we capture frame to populate the NEXT infer request
            // in the regular mode, we capture frame to the CURRENT infer request
            if (!caps[camera_index].read(source_frame)) {
                if (source_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }

            double crop_width = widths[camera_index] - cameras_cr[camera_index] - cameras_cl[camera_index];
            double crop_height = heights[camera_index] - cameras_ct[camera_index] - cameras_cb[camera_index];

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(cameras_cl[camera_index], cameras_ct[camera_index], crop_width, crop_height);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedRef(source_frame, myROI);

            // Copy the data into new matrix
            croppedRef.copyTo(frames[camera_index]);

            FrameToBlob(frames[camera_index], async_infer_request_curr[camera_index], inputName);

            t1[camera_index] = std::chrono::high_resolution_clock::now();
            ocv_decode_time[camera_index] = std::chrono::duration_cast<ms>(t1[camera_index] - t0[camera_index]).count();

            t0[camera_index] = std::chrono::high_resolution_clock::now();

            async_infer_request_curr[camera_index]->StartAsync();

            camera_index++;
            if(camera_index >= num_cameras) {
                camera_index = 0;
                all_cameras_started = true;
            }

            if(!all_cameras_started) {
                continue;
            }

            if(all_cameras_started && !alive_signal_sent) {
                mqtt::topic current_topic(*alive_topic);
                current_topic.publish(std::move("ON"));

                alive_signal_sent = true;
            }

            bool has_people_in_frame = false;

            if (OK == async_infer_request_curr[camera_index]->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1[camera_index] = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1[camera_index] - t0[camera_index]);

                t0[camera_index] = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0[camera_index] - wallclock[camera_index]);
                wallclock[camera_index] = t0[camera_index];

                t0[camera_index] = std::chrono::high_resolution_clock::now();
                std::ostringstream out;
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (ocv_decode_time[camera_index] + ocv_render_time[camera_index]) << " ms";
                cv::putText(frames[camera_index], out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                out.str("");
                out << "Wallclock time: ";
                out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                cv::putText(frames[camera_index], out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));

                out.str("");
                out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                    << " ms ("
                    << 1000.f / detection.count() << " fps)";
                cv::putText(frames[camera_index], out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                            cv::Scalar(255, 0, 0));

                // ---------------------------Processing output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
                unsigned long resized_im_h = getTensorHeight(inputDesc);
                unsigned long resized_im_w = getTensorWidth(inputDesc);
                std::vector<DetectionObject> objects;
                // Parsing outputs
                for (auto &output : outputInfo) {
                    auto output_name = output.first;
                    CNNLayerPtr layer = cnnNetwork.getLayerByName(output_name.c_str());
                    Blob::Ptr blob = async_infer_request_curr[camera_index]->GetBlob(output_name);
                    ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, heights[camera_index], widths[camera_index], FLAGS_t, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
                for (size_t i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (size_t j = i + 1; j < objects.size(); ++j)
                        if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                            objects[j].confidence = 0;
                }
                // Drawing boxes
                for (auto &object : objects) {
                    if (object.confidence < FLAGS_t)
                        continue;
                    auto label = object.class_id;
                    float confidence = object.confidence;
                    if (FLAGS_r) {
                        std::cout << "[" << label << "] element, prob = " << confidence <<
                                  "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }
                    if (confidence > FLAGS_t) {
                        if(labels[label] == std::string("person"))
                            has_people_in_frame = true;

                        /** Drawing only objects when >confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frames[camera_index],
                                (label < static_cast<int>(labels.size()) ?
                                        labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                                    cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    cv::Scalar(0, 0, 255));
                        cv::rectangle(frames[camera_index], cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                                      cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
                    }
                }
            }

            mqtt::topic current_topic(*topics[camera_index]);

            if (!FLAGS_no_show) {
                cv::imshow(camera_names[camera_index], frames[camera_index]);
            }

            if(has_people_in_frame && !humans_detected[camera_index]) {
                humans_detected[camera_index] = true;
                current_topic.publish(std::move("ON"));
            }

            if(has_people_in_frame) {
                time_humans_detected[camera_index] = std::chrono::high_resolution_clock::now();
            }

            if(!has_people_in_frame && humans_detected[camera_index]) {
                auto time_no_humans = std::chrono::high_resolution_clock::now();

                ms time_since_humans = std::chrono::duration_cast<ms>(time_no_humans - time_humans_detected[camera_index]);

                if(time_since_humans.count() > (FLAGS_to * 1000)) {
                    humans_detected[camera_index] = false;
                    current_topic.publish(std::move("OFF"));
                }
            }

            t1[camera_index] = std::chrono::high_resolution_clock::now();
            ocv_render_time[camera_index] = std::chrono::duration_cast<ms>(t1[camera_index] - t0[camera_index]).count();

            if (isLastFrame) {
                break;
            }

            const int key = cv::waitKey(1);
            if (27 == key) {  // Esc
                exit_gracefully = true;
                break;
            }
            if(exit_gracefully) {
                break;
            }
        }

        cv::destroyAllWindows();

        mqtt::topic not_alive_topic(*alive_topic);
        not_alive_topic.publish(std::move("OFF"));

        /** Showing performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr[camera_index], std::cout, getFullDeviceName(ie, FLAGS_d));
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
