#include "include/YoloDetector.h"

YoloDetector::YoloDetector(const char *labelFilePath_, const char *modelPath_, const char *testImagePath_) {
    this->labelFilePath = labelFilePath_;
    this->modelPath = modelPath_;
    this->testImagePath = testImagePath_;
}
