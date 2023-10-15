#ifndef MY_APPLICATION_FFMPEG_PLAYER_KT_UTIL_H
#define MY_APPLICATION_FFMPEG_PLAYER_KT_UTIL_H

#include <android/log.h>

#define THREAD_MAIN 1
#define THREAD_CHILD 2

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "KTPlayer", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "KTPlayer", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "KTPlayer", __VA_ARGS__)


#define FFMPEG_CAN_NOT_OPEN_URL 1
#define FFMPEG_CAN_NOT_FIND_STREAMS 2
#define FFMPEG_FIND_DECODER_FAIL 3
#define FFMPEG_ALLOC_CODEC_CONTEXT_FAIL 4
#define FFMPEG_CODEC_CONTEXT_PARAMETERS_FAIL 6
#define FFMPEG_OPEN_DECODER_FAIL 7
#define FFMPEG_NO_MEDIA 8

#define NMS_THRESH        0.6
#define BOX_THRESH        0.5


#endif //MY_APPLICATION_FFMPEG_PLAYER_KT_UTIL_H
