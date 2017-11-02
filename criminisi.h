#pragma once

#include <opencv2/core/core.hpp>

#include <vector>
#include <set>
#include <utility>
#include <map>

class Criminisi
{
public:
    static constexpr double DEFAULT_W = 0.1;
    static constexpr double DEFAULT_DELTA = 0.00005;
    
public:
    Criminisi (const cv::Mat& image, const int window_radius = 4);
    
    Criminisi (cv::Mat&& image, const int window_radius = 4);
    
public:
    void mask (const cv::Mat_<bool>& m) { _mask = m; }
    void mask (cv::Mat_<bool>&& m)      { _mask = m; }
    
    const cv::Mat& mask (void) { return _mask; }
    
public:
    int     radius  (void) { return _radius; }
    double  w       (void) { return _w; }
    double  delta   (void) { return _delta; }
    
    void    radius  (double r) { _radius = r; }
    void    w       (double u) { _w = u; }
    void    delta   (double d) { _delta = d; }
    
public:
    cv::Mat generate (void);

    void draw_contour (cv::Mat& image, const cv::Vec3b& colour);
    void draw_contour (cv::Mat& image, const uchar colour);

private:
    void generate_contour (void);
    void generate_priority (void);
    
    cv::Point2d generate_normal (const cv::Point& p, int radius);
    
    void update_contour (const cv::Point& p);
    
    double priority (const std::pair<int, int>& p);
    
private:
    cv::Mat patch (const cv::Point& p, const cv::Mat& img);
    cv::Mat patch (const cv::Point& p, const cv::Mat& img, const int radius);
    
protected:
    cv::Mat _original;
    cv::Mat _modified;
    cv::Mat_<bool> _mask;
    
    cv::Mat _confidence;
        
protected:
    std::set<std::pair<int, int>> _contour;
    
    std::set<std::pair<double, std::pair<int, int>>> _pq;
    std::map<std::pair<int, int>, double> _map;
    
protected:
    int _rows;
    int _cols;
    
    int _radius;
    
protected:
    double _w;
    double _delta;
};