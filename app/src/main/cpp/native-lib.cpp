#include <jni.h>
#include <string>
#include "util.h"
#include "rknn_api.h"
#include "postprocess.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myarm64rknn_MainActivity_stringFromJNI(JNIEnv *env, jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
    unsigned char *data;
    int ret;

    data = nullptr;

    if (nullptr == fp) {
        return nullptr;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        LOGD("blob seek failure.\n");
        return nullptr;
    }

    data = (unsigned char *) malloc(sz);
    if (data == nullptr) {
        LOGD("buffer malloc failure.\n");
        return nullptr;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

unsigned char *load_data_file(const char *filename, int *model_size) {

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        LOGD("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

const char *get_type_string(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32:
            return "FP32";
        case RKNN_TENSOR_FLOAT16:
            return "FP16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        default:
            return "UNKNOW";
    }
}

const char *get_format_string(rknn_tensor_format fmt) {
    switch (fmt) {
        case RKNN_TENSOR_NCHW:
            return "NCHW";
        case RKNN_TENSOR_NHWC:
            return "NHWC";
        default:
            return "UNKNOW";
    }
}

#define IMAGE_WIDTH  640
#define IMAGE_HEIGHT 640
#define NUM_CHANNELS 3

// 计算一维数组中元素的索引
int index(int c, int h, int w) {
    return c * IMAGE_HEIGHT * IMAGE_WIDTH + h * IMAGE_WIDTH + w;
}

// 转换函数
void convertToNCHW(unsigned char *input_data, unsigned char *output_data) {
    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int h = 0; h < IMAGE_HEIGHT; h++) {
            for (int w = 0; w < IMAGE_WIDTH; w++) {
                int input_index = index(c, h, w);
                int output_index = index(c, h, w);
                output_data[output_index] = input_data[input_index];
            }
        }
    }
}

const char *get_qnt_type_string(rknn_tensor_qnt_type type) {
    switch (type) {
        case RKNN_TENSOR_QNT_NONE:
            return "NONE";
        case RKNN_TENSOR_QNT_DFP:
            return "DFP";
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
            return "AFFINE";
        default:
            return "UNKNOW";
    }
}

void dump_tensor_attr(rknn_tensor_attr *attr) {
    LOGD("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myarm64rknn_YoloDetector_startDetect(JNIEnv *env,
                                                      jobject thiz,
                                                      jstring image_data_path,
                                                      jstring model_path,
                                                      jstring label_path) {

    const char *image_data_source_ = env->GetStringUTFChars(image_data_path, nullptr);
    const char *model_path_ = env->GetStringUTFChars(model_path, nullptr);
    const char *label_path_ = env->GetStringUTFChars(label_path, nullptr);

    LOGD("model_data_source_: %s", image_data_source_);
    LOGD("model_path_: %s", model_path_);
    LOGD("label_path_: %s", label_path_);

    rknn_context ctx;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    int ret;

    LOGD("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

    LOGD("Loading model...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_data_file(model_path_, &model_data_size);
    if (model_data == nullptr) {
        LOGD("load model failed!\n");
        return;
    }
    LOGD("rknn_init...\n");
    ret = rknn_init(&ctx, model_data, model_data_size, RKNN_FLAG_PRIOR_MEDIUM | RKNN_FLAG_ASYNC_MASK);
    if (ret < 0) {
        LOGD("rknn_init error ret=%d\n", ret);
        return;
    }

    LOGD("checking version ");
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        LOGD("rknn_init error ret=%d\n", ret);
        return;
    }
    LOGD("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        LOGD("rknn_init error ret=%d\n", ret);
        return;
    }

    LOGD("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOGD("rknn_init error ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOGD("rknn_init error ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&(output_attrs[i]));
        if (output_attrs[i].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || output_attrs[i].type != RKNN_TENSOR_UINT8) {
            LOGD("The Demo required for a Affine asymmetric u8 quantized rknn model, but output quant type is %s, output data type is %s\n",
                 get_qnt_type_string(output_attrs[i].qnt_type), get_type_string(output_attrs[i].type));
            return;
        }
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        LOGD("model is NCHW input fmt\n");
        width = input_attrs[0].dims[0];
        height = input_attrs[0].dims[1];
    } else {
        LOGD("model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
    }

    LOGD("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    LOGD("Loading image...\n");
    int imageSize = 0;
    unsigned char *imageData = load_data_file(image_data_source_, &imageSize);
    LOGD("imageSize: %d", imageSize);
    if (imageData == nullptr) {
        LOGD("load image failed!\n");
        return;
    }
    unsigned char *imageDataNCHW = (unsigned char *) malloc(imageSize);
    convertToNCHW(imageData, imageDataNCHW);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = imageData;

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        LOGD("rknn_run error ret=%d\n", ret);
        return;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) {
        LOGD("rknn_outputs_get error ret=%d\n", ret);
        return;
    }
    gettimeofday(&stop_time, NULL);
    LOGD("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    float scale_w = 1.0f;
    float scale_h = 1.0f;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process((uint8_t *) outputs[0].buf,
                 (uint8_t *) outputs[1].buf,
                 (uint8_t *) outputs[2].buf,
                 height, width,
                 box_conf_threshold, nms_threshold,
                 scale_w, scale_h, out_zps, out_scales,
                 &detect_result_group, label_path_);

    env->ReleaseStringUTFChars(image_data_path, image_data_source_);
    env->ReleaseStringUTFChars(model_path, model_path_);
    env->ReleaseStringUTFChars(label_path, label_path_);

    LOGD("detect finished");
    // 释放资源
    rknn_destroy(ctx);
    free(model_data);

}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_myarm64rknn_YoloDetector_prepare(JNIEnv *env, jobject thiz) {
    // TODO: implement prepare()
}