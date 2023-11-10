#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

// 모델 3개 이용
static NanoDet* g_nanodet = 0;
static ncnn::Mutex lock;

bool setBitmapFromMat(JNIEnv *pEnv, jobject pJobject, cv::Mat mat);

jobject MatToBitmap(JNIEnv *env, cv::Mat src){
    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888FieldID = env->GetStaticFieldID(bitmapConfigClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject argb8888Config = env->GetStaticObjectField(bitmapConfigClass, argb8888FieldID);

    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");

    jobject bitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, src.cols, src.rows, argb8888Config);

    AndroidBitmapInfo bitmapInfo;
    void* pixels = 0;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
        return NULL;
    }

    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return NULL;
    }

    cv::Mat dst(bitmapInfo.height, bitmapInfo.width, CV_8UC4, pixels);

    if (src.type() == CV_8UC3) {
        cv::cvtColor(src, dst, cv::COLOR_RGB2RGBA);
    } else if (src.type() == CV_8UC1) {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2RGBA);
    } else {
        src.copyTo(dst);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    return bitmap;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_nanodet;
        g_nanodet = 0;
    }
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu /*&& ncnn::get_gpu_count()*/ == 0)
        {
            // no gpu
            delete g_nanodet;
            g_nanodet = 0;
        }
        else
        {
            if (!g_nanodet)
                g_nanodet = new NanoDet;
            g_nanodet->load(mgr, use_gpu);
        }
    }
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_predict(JNIEnv *env, jobject thiz, jobject imageView, jobject bitmap) {

    // RGB형식으로 변경
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        return nullptr;
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return nullptr;
    }

    // Get the pointer to bitmap pixels
    void* bitmapPixels;
    if (AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) < 0) {
        return nullptr;
    }

    // Create a cv::Mat from the bitmap data
    int width = info.width;
    int height = info.height;
    cv::Mat rgba(height, width, CV_8UC4, bitmapPixels);
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);

//    cv::resize(rgb,rgb,cv::Size(640,640));

    // 이미지 뷰 업데이트 JNI 호출
    jclass imageViewClass = env->GetObjectClass(imageView);
    jmethodID setImageBitmapMethod = env->GetMethodID(imageViewClass, "setImageBitmap", "(Landroid/graphics/Bitmap;)V");

    ncnn::MutexLockGuard g(lock);

    if (g_nanodet)
    {
        std::string result;
        __android_log_print(ANDROID_LOG_ERROR, "Detect","Start Detect");
        g_nanodet->detect(rgb);
        __android_log_print(ANDROID_LOG_ERROR, "Detect","End Detect");

        // 화면 크기에 맞게 조정
        jclass viewClass = env->GetObjectClass(imageView);
        jmethodID getWidthMethodID = env->GetMethodID(viewClass, "getWidth", "()I");
        jmethodID getHeightMethodID = env->GetMethodID(viewClass, "getHeight", "()I");

        jint viewWidth = env->CallIntMethod(imageView, getWidthMethodID);
        jint viewHeight = env->CallIntMethod(imageView, getHeightMethodID);

        float screenRatio = static_cast<float>(viewWidth) / viewHeight;
        float imageRatio = static_cast<float>(width) / height;

        int resizeWidth, resizeHeight;

        if (imageRatio > screenRatio) {
            resizeWidth = viewWidth;
            resizeHeight = static_cast<int>(viewWidth / imageRatio);
        } else {
            resizeHeight = viewHeight;
            resizeWidth = static_cast<int>(viewHeight * imageRatio);
        }

        // 객체 감지 후 이미지 크기 조정
        cv::resize(rgb, rgb, cv::Size(resizeWidth, resizeHeight));

        // 자바로 반환, imageView 나타내기
        jobject jbitmap = MatToBitmap(env, rgb);
        env->CallVoidMethod(imageView, setImageBitmapMethod, jbitmap);

        // BBox값 문자열로 나타내기
        __android_log_print(ANDROID_LOG_ERROR, "Data : ","%s", result.c_str());
        // C++ 문자열을 Java 문자열로 변환
        jstring jResult = env->NewStringUTF(result.c_str());

        return jResult;

    }else{
        draw_unsupported(rgb);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    //return JNI_TRUE;
    return nullptr;
}
}