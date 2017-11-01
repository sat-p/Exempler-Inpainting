#include "criminisi.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

/*****************************************************************************/

Criminisi::Criminisi (const cv::Mat& image, const int window_radius) :
    _original (image),
    _rows     (image.rows),
    _cols     (image.cols),
    _radius   (window_radius)
{}

/*****************************************************************************/

Criminisi::Criminisi (cv::Mat&& image, const int window_radius) :
    _original (image),
    _rows     (image.rows),
    _cols     (image.cols),
    _radius   (window_radius)
{}

/*****************************************************************************/

cv::Mat Criminisi::generate (void)
{
    cv::Sobel (_original, _dx, -1, 1, 0);
    cv::Sobel (_original, _dy, -1, 0, 1);
    
    generate_contour();
    _modified = _original.clone().setTo (cv::Vec3b (0, 0, 0),
                                         _mask);
    
    generate_priority();
    
    cv::imshow ("mask", _mask);
    
    while (_pq.size()) {
    
        const std::pair<int, int>& point = _pq.rbegin()->second;
        const cv::Point p (point.first, point.second);
        
        const auto& phi_p = patch (p, _modified);
        
        cv::Mat templateMask = (patch (p, _mask) != 0);
        cv::Mat mergeArrays[3] = {templateMask, templateMask, templateMask};
        cv::merge(mergeArrays, 3, templateMask);
        
        cv::Mat res;
        cv::matchTemplate (_modified, phi_p, res, CV_TM_SQDIFF, templateMask);
        
        std::cerr << "Points in contour : " << _pq.size() << std::endl;

        assert (_map[point] == _pq.rbegin()->first);
        
        cv::Mat dilatedMask;

        cv::dilate (_mask, dilatedMask, cv::Mat(), cv::Point (-1, -1), _radius);
        
        res.setTo (std::numeric_limits<float>::max(),
                   dilatedMask (cv::Range (0, res.rows),
                                cv::Range (0, res.cols)));
        
        cv::Point q_;
        cv::minMaxLoc(res, NULL, NULL, &q_);
        
        const int x_begin = std::max (p.x - _radius, 0);
        const int y_begin = std::max (p.y - _radius, 0);
        
        const int dx = p.x - x_begin;
        const int dy = p.y - y_begin;
        
        const cv::Point& q = q_ + cv::Point (dx, dy);
        assert (p != q);
        
        const auto& phi_q = patch (q, _modified);
        
        cv::Mat p1, q1, t1;
        cv::resize (phi_p, p1, cv::Size (100, 100));
        cv::resize (phi_q, q1, cv::Size (100, 100));
        cv::resize (templateMask, t1, cv::Size (100, 100));     
        cv::namedWindow ("phi_p : ");
        cv::namedWindow ("phi_q : ");
        cv::imshow ("phi_p : ", p1);
        cv::imshow ("phi_q : ", q1);
        cv::imshow ("t1 : ", t1);
        cv::imshow ("dilated mask", dilatedMask);
        cv::imshow ("mask", _mask);
        cv::imshow ("modified", _modified);
        cv::imshow ("confidence", _confidence);
        
        cv::Mat p_mask = patch (p, _mask);        
        phi_q.copyTo (phi_p, p_mask);
        
        cv::Mat confidencePatch = patch (p, _confidence);
        
        const double confidence = cv::sum (confidencePatch)[0] /
                                  confidencePatch.total();
        confidencePatch.setTo (confidence, p_mask);
                                  
        cv::waitKey (10);
        
        p_mask.setTo (0);
        
        update_contour (p);
    }
    
    assert (cv::countNonZero (_mask) == 0);
    
    return _modified;
}

/*****************************************************************************/

void Criminisi::draw_contour (cv::Mat& image, const cv::Vec3b& colour)
{
    for (const auto& c : _contour)
        image.at<cv::Vec3b> (c.second, c.first) = colour;
}

void Criminisi::draw_contour (cv::Mat& image, const uchar colour)
{
    for (const auto& c : _contour)
        image.at<uchar> (c.second, c.first) = colour;
}

/*****************************************************************************/

void Criminisi::generate_contour (void)
{
    constexpr int NUM_N = 8;
    
    const int dx8[NUM_N] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[NUM_N] = { 0, -1, -1, -1, 0, 1, 1,  1};
    
    _contour.clear();
    
    for (int i = 0; i < _cols; ++i) {
        for (int j = 0; j < _rows; ++j) {
            for (int k = 0; k < NUM_N; ++k) {
                
                if (!_mask.at<uchar> (j, i))
                    continue;
                
                const int x = i + dx8[k];
                const int y = j + dy8[k];
                
                if (x >= 0 && x < _cols && y >= 0 && y < _rows) {
                
                    if (!_mask.at<uchar> (y, x)) {
                    
                        _contour.emplace (i, j);
                        break;
                    }
                }
            }
        }
    }
}

/*****************************************************************************/

void Criminisi::generate_priority (void)
{
    _confidence = cv::Mat::zeros (_rows, _cols, CV_64FC1);
    
    for (int i = 0; i < _cols; ++i)
        for (int j = 0; j < _rows; ++j)
            _confidence.at<double> (j, i) = !_mask.at<uchar> (j, i);
    
//     while (_pq.size())
//         _pq.pop();
    _pq.clear();
    _map.clear();
    
    for (const auto& c : _contour) {
        const double pri = priority (c);
        _pq.emplace (pri, c);
        _map[c] = pri;
    }

}

/*****************************************************************************/

cv::Point2d Criminisi::generate_normal (const int x_begin, const int y_begin,
                                        const int x_end,   const int y_end)
{
    std::vector<double> X;
    std::vector<double> Y;
    
    for (int i = x_begin; i <= x_end; ++i) {
        for (int j = y_begin; j <= y_end; ++j) {
            if (_contour.count (std::make_pair (i, j))) {
            
                X.push_back (i);
                Y.push_back (j);
            }
        }
    }
    
    if (X.size() == 1)
        return cv::Point2d (1.0, 0);
    
    cv::Mat X_ (cv::Size (2, X.size()), CV_64FC1);
    cv::Mat Y_ (cv::Size (1, X.size()), CV_64FC1);
    
    for (int i = 0; i < (int) X.size(); ++i) {
    
        auto* x = X_.ptr<double> (i);
        auto* y = Y_.ptr<double> (i);
        
        x[0] = X[i];
        x[1] = 1.0;
        
        y[0] = Y[i];
    }
    
    cv::Mat sol;
    
    cv::solve (X_, Y_, sol, cv::DECOMP_SVD);
    
    assert (sol.type() == CV_64F);
    
    const float slope = sol.at<float> (0);
    cv::Point2d normal (-slope, 1);
    
    if (std::isnan (slope))
        normal = cv::Point2d (1, 0);
    
    return normal / cv::norm (normal);
}

/*****************************************************************************/

void Criminisi::update_contour (const cv::Point& p)
{
    constexpr int NUM_N = 8;

    const int dx8[NUM_N] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[NUM_N] = { 0, -1, -1, -1, 0, 1, 1,  1};
    
    const int x_begin = std::max (p.x - 2 *_radius, 0);
    const int y_begin = std::max (p.y - 2 *_radius, 0);
    const int x_end = std::min (p.x + 2 *_radius, _cols - 1);
    const int y_end = std::min (p.y + 2 *_radius, _rows - 1);
    
    for (int i = x_begin; i <= x_end; ++i) {
        for (int j = y_begin; j <= y_end; ++j) {
            
            const std::pair<int, int> p = std::make_pair (i, j);
            
            if (_contour.count (p)) {
                const double pri = _map[p];
                _contour.erase (p);
                _pq.erase (std::make_pair (pri, p));
                _map.erase (p);
            }
        }
    }
    
    std::set<std::pair<int, int>> new_contour;
    
    for (int i = x_begin; i <= x_end; ++i) {
        for (int j = y_begin; j <= y_end; ++j) {
            for (int k = 0; k < NUM_N; ++k) {
                
                if (!_mask.at<uchar> (j, i))
                    continue;
                
                const int x = i + dx8[k];
                const int y = j + dy8[k];
                
                if (x >= 0 && x < _cols && y >= 0 && y < _rows) {
                
                    if (!_mask.at<uchar> (y, x)) {
                    
                        new_contour.emplace (i, j);
                        break;
                    }
                }
            }
        }
    }
    
    for (const auto& nc : new_contour)
        _contour.emplace (nc);
    
    for (const auto& nc : new_contour) {
        
        const double pri = priority (nc);
        _pq.emplace (std::make_pair (pri, nc));
        _map[nc] = pri;
    }
}

/*****************************************************************************/ 

double Criminisi::priority (const std::pair<int, int>& p)
{
    const cv::Mat& confidencePatch = patch (cv::Point (p.first, p.second),
                                            _mask);
    
    assert (confidencePatch.total());
    
    double confidence = cv::sum (confidencePatch)[0] /
                        confidencePatch.total();
                        
    const int x_begin = std::max (p.first - _radius, 0);
    const int y_begin = std::max (p.second - _radius, 0);
    const int x_end = x_begin + confidencePatch.cols;
    const int y_end = y_begin + confidencePatch.rows;
    
    const auto& normal = generate_normal (x_begin, y_begin, x_end, y_end);
    
    const cv::Vec3d& dx = _dx.at<cv::Vec3b> (p.second, p.first);
    const cv::Vec3d& dy = _dy.at<cv::Vec3b> (p.second, p.first);
    
    const cv::Vec3d& dot = dx * normal.x + dy * normal.y;
    
    return confidence * cv::norm (dot);
}

/*****************************************************************************/ 

cv::Mat Criminisi::patch (const cv::Point& p, const cv::Mat& img)
{
    const int x_begin = std::max (p.x - _radius, 0);
    const int y_begin = std::max (p.y - _radius, 0);
    
    const int x_end = std::min (p.x + _radius + 1, _cols + 1);
    const int y_end = std::min (p.y + _radius + 1, _rows + 1);
    
    return img  (cv::Range (y_begin, y_end),
                 cv::Range (x_begin, x_end));
}

/*****************************************************************************/
/*****************************************************************************/