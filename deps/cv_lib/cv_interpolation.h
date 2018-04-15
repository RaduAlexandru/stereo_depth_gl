#ifndef __OPENCV_TEST_INTERPOLATION_HPP__
#define __OPENCV_TEST_INTERPOLATION_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

template <typename T> T readVal(const cv::Mat& src, int y, int x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
{
    if (border_type == cv::BORDER_CONSTANT)
        return (y >= 0 && y < src.rows && x >= 0 && x < src.cols) ? src.at<T>(y, x * src.channels() + c) : cv::saturate_cast<T>(borderVal.val[c]);

    return src.at<T>(cv::borderInterpolate(y, src.rows, border_type), cv::borderInterpolate(x, src.cols, border_type) * src.channels() + c);
}

template <typename T> struct NearestInterpolator
{
    static T getValue(const cv::Mat& src, float y, float x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
    {
        return readVal<T>(src, int(y), int(x), c, border_type, borderVal);
    }
};

template <typename T> struct LinearInterpolator
{
    static T getValue(const cv::Mat& src, float y, float x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
    {
        int x1 = cvFloor(x);
        int y1 = cvFloor(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        float res = 0;

        res += readVal<T>(src, y1, x1, c, border_type, borderVal) * ((x2 - x) * (y2 - y));
        res += readVal<T>(src, y1, x2, c, border_type, borderVal) * ((x - x1) * (y2 - y));
        res += readVal<T>(src, y2, x1, c, border_type, borderVal) * ((x2 - x) * (y - y1));
        res += readVal<T>(src, y2, x2, c, border_type, borderVal) * ((x - x1) * (y - y1));

        return cv::saturate_cast<T>(res);
    }
};

template <typename T> struct CubicInterpolator
{
    static float bicubicCoeff(float x_)
    {
        float x = fabsf(x_);
        if (x <= 1.0f)
        {
            return x * x * (1.5f * x - 2.5f) + 1.0f;
        }
        else if (x < 2.0f)
        {
            return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
        }
        else
        {
            return 0.0f;
        }
    }

    static T getValue(const cv::Mat& src, float y, float x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
    {
        const float xmin = ceilf(x - 2.0f);
        const float xmax = floorf(x + 2.0f);

        const float ymin = ceilf(y - 2.0f);
        const float ymax = floorf(y + 2.0f);

        float sum  = 0.0f;
        float wsum = 0.0f;

        for (float cy = ymin; cy <= ymax; cy += 1.0f)
        {
            for (float cx = xmin; cx <= xmax; cx += 1.0f)
            {
                const float w = bicubicCoeff(x - cx) * bicubicCoeff(y - cy);
                sum += w * readVal<T>(src, (int) floorf(cy), (int) floorf(cx), c, border_type, borderVal);
                wsum += w;
            }
        }

        float res = (!wsum)? 0 : sum / wsum;

        return cv::saturate_cast<T>(res);
    }
};

#endif // __OPENCV_TEST_INTERPOLATION_HPP__