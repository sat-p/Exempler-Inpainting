#pragma once

#include<opencv2/core/core.hpp>

class Criminisi
{
public:
    Criminisi (const cv::Mat& image);
    
    Criminisi (cv::Mat&& image);
    
public:
    void mask (const cv::Mat& m)
    { _mask = m; }
    
    void mask (cv::Mat&& m)
    { _mask = m; }
    
    const cv::Mat& mask (void)
    { return _mask; }
    
public:
    cv::Mat generate_inpainted (void);
    
protected:
    cv::Mat _original;
    cv::Mat _mask;
};