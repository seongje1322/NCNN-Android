#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>
#include <android/log.h>
#include <net.h>

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

class NanoDet
{
public:
    NanoDet();

    int load(AAssetManager* mgr, bool use_gpu = false);

    int detect(cv::Mat& rgb);

private:
    ncnn::Net yolop;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H
