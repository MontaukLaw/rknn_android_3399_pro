#ifndef MYARM64RKNN_YOLODETECTOR_H
#define MYARM64RKNN_YOLODETECTOR_H


class YoloDetector {

private:
    const char *labelFilePath = 0;
    const char *modelPath = 0;
    const char *testImagePath= 0;
public:

    YoloDetector(const char *labelFilePath_, const char *modelPath_, const char *testImagePath_);

};


#endif //MYARM64RKNN_YOLODETECTOR_H
