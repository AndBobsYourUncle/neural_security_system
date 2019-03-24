#include <stdlib.h>     /* getenv */
#include <string>
#include <chrono>
#include <gflags/gflags.h>

#include <random>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <ie_device.hpp>
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>
#include <ie_plugin_cpp.hpp>
#include <ie_extension.h>
#include "ext_list.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <sstream>
#include <stdio.h>
#include <string.h>
#include <termios.h>

#include <csignal>

#include "main.hpp"
#include "slog.hpp"
#include "args_helper.hpp"
#include "common.hpp"
#include "ocv_common.hpp"

#include "mqtt/async_client.h"

using namespace std::chrono;

using namespace std;
using namespace cv;
using namespace InferenceEngine::details;
using namespace InferenceEngine;


const int    QOS = 1;

const auto PERIOD = seconds(5);

const int MAX_BUFFERED_MSGS = 120;  // 120 * 5sec => 10min off-line buffering

const string PERSIST_DIR { "data-persist" };

bool exit_gracefully = false;

#define yolo_scale_13 13
#define yolo_scale_26 26
#define yolo_scale_52 52

void signalHandler(int signum) {
   cout << "Interrupt signal (" << signum << ") received. Exiting gracefully...\n";

   exit_gracefully = true;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
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

    if (FLAGS_tp.empty()) {
        throw std::logic_error("Parameter -tp is not set");
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

    bool operator<(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
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
    try { num = layer->GetParamAsInts("mask").size(); } catch (...) {}
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    int anchor_offset = 0;

    //throw std::runtime_error("anchors.size() ==" + std::to_string(anchors.size()));

    if (anchors.size() == 18) {        // YoloV3
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 6;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_52:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    } else if (anchors.size() == 12) { // tiny-YoloV3
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    } else {                           // ???
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 6;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_52:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    }
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
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
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

    // ------------------------------ Parsing and validating the input arguments ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    string address = FLAGS_mh;

    mqtt::async_client cli(address, "", MAX_BUFFERED_MSGS, PERSIST_DIR);

    mqtt::connect_options connOpts;
    connOpts.set_keep_alive_interval(MAX_BUFFERED_MSGS * PERIOD);
    connOpts.set_clean_session(true);
    connOpts.set_automatic_reconnect(true);

    std::cout << "MQTT Username: " << FLAGS_d << std::endl;

    connOpts.set_user_name(FLAGS_u);
    connOpts.set_password(FLAGS_p);

    // Create a topic object. This is a conventience since we will
    // repeatedly publish messages with the same parameters.
    mqtt::topic top(cli, FLAGS_tp, QOS, true);

    // Random number generator [0 - 100]
    random_device rnd;
    mt19937 gen(rnd());
    uniform_int_distribution<> dis(0, 100);

    try {
        // Connect to the MQTT broker
        cout << "Connecting to server '" << address << "'..." << flush;
        cli.connect(connOpts)->wait();
        cout << "OK\n" << endl;

        /** This demo covers a certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (FLAGS_i == "cam0") {
            cap.open(0);
        } else if (FLAGS_i == "cam1") {
            cap.open(1);
        } else if (FLAGS_i == "cam2") {
            cap.open(2);
        } else if (!(cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        // // read input (video) frame
        cv::Mat frame;  cap >> frame;
        cv::Mat next_frame;

        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed to get next frame from the " + FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({"../lib", ""}).getPluginByDevice(FLAGS_d);
        printPluginVersion(plugin, std::cout);

        /**Loading extensions to the plugin **/

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from the "extension" folder containing
             * custom CPU layer implementations.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            plugin.AddExtension(extension_ptr);
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Reading network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Setting batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extracting the model name and loading its weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
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
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
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
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        //if (outputInfo.size() != 3) {
        //    throw std::logic_error("This demo only accepts networks with three layers");
        //}
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        ExecutableNetwork network = plugin.LoadNetwork(netReader.getNetwork(), {});

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Creating infer request -----------------------------------------------
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_decode_time = 0, ocv_render_time = 0;

        auto time_humans_detected = std::chrono::high_resolution_clock::now();
        bool humans_detected = false;

        while (true) {
            cv::Mat source_frame;

            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the Async mode, we capture frame to populate the NEXT infer request
            // in the regular mode, we capture frame to the CURRENT infer request
            if (!cap.read(source_frame)) {
                if (source_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }

            double crop_width = width - FLAGS_cr - FLAGS_cl;
            double crop_height = height - FLAGS_ct - FLAGS_cb;

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(FLAGS_cl, FLAGS_ct, crop_width, crop_height);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedRef(source_frame, myROI);

            // Copy the data into new matrix
            croppedRef.copyTo(next_frame);

            if (isAsyncMode) {
                if (isModeChanged) {
                    FrameToBlob(frame, async_infer_request_curr, inputName);
                }
                if (!isLastFrame) {
                    FrameToBlob(next_frame, async_infer_request_next, inputName);
                }
            } else if (!isModeChanged) {
                FrameToBlob(frame, async_infer_request_curr, inputName);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
            // in the regular mode, we start the CURRENT request and wait for its completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }

            bool has_people_in_frame = false;

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();
                std::ostringstream out;
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (ocv_decode_time + ocv_render_time) << " ms";
                cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                out.str("");
                out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                cv::putText(frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                    out.str("");
                    out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                }

                // ---------------------------Processing output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                unsigned long resized_im_h = inputInfo.begin()->second.get()->getDims()[0];
                unsigned long resized_im_w = inputInfo.begin()->second.get()->getDims()[1];
                std::vector<DetectionObject> objects;
                // Parsing outputs
                for (auto &output : outputInfo) {
                    auto output_name = output.first;
                    //slog::info << "output_name = " + output_name << slog::endl;
                    CNNLayerPtr layer = netReader.getNetwork().getLayerByName(output_name.c_str());
                    Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                    ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end());
                for (int i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (int j = i + 1; j < objects.size(); ++j) {
                        if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t) {
                            objects[j].confidence = 0;
                        }
                        //if (objects[j].confidence == 1) {
                        //    objects[j].confidence = 0;
                        //}
                    }
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
                    //slog::info << "confidence = " + std::to_string(confidence) << slog::endl;
                    if (confidence > FLAGS_t) {
                        if(labels[label] == std::string("person"))
                            has_people_in_frame = true;

                        /** Drawing only objects when >confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frame,
                                (label < labels.size() ? labels[label] : std::string("label #") + std::to_string(label))
                                    + conf.str(),
                                    cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                        cv::rectangle(frame, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }
                }
            }

            if(!FLAGS_no_image)
                cv::imshow("Detection results", frame);

            if(has_people_in_frame && !humans_detected) {
                humans_detected = true;
                top.publish(std::move("ON"));
            }

            if(has_people_in_frame) {
                time_humans_detected = std::chrono::high_resolution_clock::now();
            }

            if(!has_people_in_frame && humans_detected) {
                auto time_no_humans = std::chrono::high_resolution_clock::now();

                ms time_since_humans = std::chrono::duration_cast<ms>(time_no_humans - time_humans_detected);

                if(time_since_humans.count() > (FLAGS_to * 1000)) {
                    humans_detected = false;
                    top.publish(std::move("OFF"));
                }
            }

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }


            // Final point:
            // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
            frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            }
            if(FLAGS_no_image) {
                char kp = getchar();
                if (kp == 27) {
                    slog::info << "Key press: " << kp << slog::endl;
                    break;
                }
            }
            if(exit_gracefully)
                break;
        }
        // -----------------------------------------------------------------------------------------------------
        // auto total_t1 = std::chrono::high_resolution_clock::now();
        // ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        // std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Showing performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr, std::cout);
        }

        // Disconnect
        cout << "\nDisconnecting..." << flush;
        cli.disconnect()->wait();
        cout << "OK" << endl;
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << exc.what() << endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}