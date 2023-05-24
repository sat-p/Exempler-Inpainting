#include "criminisi.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

/*****************************************************************************/

Criminisi::Criminisi (const cv::Mat& image, const int window_radius) :
    _original (image),
    _rows     (image.rows),
    _cols     (image.cols),
    _radius   (window_radius),
    _w        (DEFAULT_W),
    _delta    (DEFAULT_DELTA)
{}

/*****************************************************************************/

Criminisi::Criminisi (cv::Mat&& image, const int window_radius) :
    _original (image),
    _rows     (image.rows),
    _cols     (image.cols),
    _radius   (window_radius),
    _w        (DEFAULT_W),
    _delta    (DEFAULT_DELTA)
{}

/*****************************************************************************/

cv::Mat Criminisi::generate (void)
{
    generate_contour();
    _modified = _original.clone().setTo (cv::Vec3b (0, 0, 0),
                                         _mask);
    
    generate_priority();
    
    cv::Mat resSSD;
    cv::Mat pMask;
    cv::Mat pInvMask;
    cv::Mat templateMask;
    cv::Mat mergeArrays[3];
    
    cv::Mat dilatedMask;
    
    while (_pq.size()) {
        
        const std::pair<int, int>& point = _pq.rbegin()->second;
        const cv::Point p (point.first, point.second);
        
        const auto& phi_p = patch (p, _modified);
        const int radius = (phi_p.rows - 1) / 2;
        
        pMask = patch (p, _mask, radius);
        pInvMask = ~pMask;
        
        templateMask = (pInvMask);
        
        for (int i = 0; i < 3; ++i)
            mergeArrays[i] = templateMask;
        
        cv::merge(mergeArrays, 3, templateMask);

        cv::matchTemplate (_modified, phi_p, resSSD, cv::TM_SQDIFF, templateMask);
        
        cv::dilate (_mask, dilatedMask, cv::Mat(), cv::Point (-1, -1), radius);
        
//         cv::Mat mean_p, var_p;
//         cv::meanStdDev (phi_p, mean_p, var_p, pInvMask);
        
//         cv::Mat mean_q, var_q;
//         for (int i = radius; i < _cols - radius; ++i) {
//             for (int j = radius; j < _rows - radius; ++j) {
//                 
//                 const cv::Point currentPoint (i, j);
//                 const cv::Point& resSSDPoint = currentPoint -
//                                             cv::Point (radius, radius);
//                 
//                 if (dilatedMask.at<uchar> (currentPoint))
//                     continue;
//                 
//                 cv::meanStdDev (patch (currentPoint, _modified, radius),
//                                 mean_q, var_q, pInvMask);
//                 
//                 resSSD.at<double> (resSSDPoint) += _delta *
//                                             std::pow (cv::norm (var_p - var_q), 2);
//             }
//         }
        
        std::cerr << "Points in contour : " << _pq.size() << std::endl;
        
        resSSD.setTo (std::numeric_limits<float>::max(),
                   dilatedMask (cv::Range (radius, _rows - radius),
                                cv::Range (radius, _cols - radius)));
        
        cv::Point q;
        cv::minMaxLoc (resSSD, NULL, NULL, &q);
        
        q = q + cv::Point (radius, radius);
        
        const auto& phi_q = patch (q, _modified, radius);
        
//         cv::Mat PHI_p, PHI_q;
//         cv::resize (phi_p, PHI_p, cv::Size (100, 100));
//         cv::resize (phi_q, PHI_q, cv::Size (100, 100));
//         
//         cv::imshow ("phi_p", PHI_p);
//         cv::imshow ("phi_q", PHI_q);
//         
        phi_q.copyTo (phi_p, pMask);
        
        cv::Mat confPatch = patch (p, _confidence);
        const double confidence = cv::sum (confPatch)[0] /
                                  confPatch.total();
        confPatch.setTo (confidence, pMask);
        
        pMask.setTo (0);
        
        update_contour (p);
        
//         cv::imshow ("modified", _modified);
//         cv::imshow ("confidence", _confidence);
//         cv::waitKey (50);
    }
    
    std::cerr << "Completed" << std::endl;
    
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
    
    _pq.clear();
    _map.clear();
    
    for (const auto& c : _contour) {
        const double pri = priority (c);
        _pq.emplace (pri, c);
        _map[c] = pri;
    }

}

/*****************************************************************************/

cv::Point2d Criminisi::generate_normal (const cv::Point& p, const int radius)
{
    std::vector<double> X;
    std::vector<double> Y;
    
    for (int i = p.x - radius; i <= p.x + radius; ++i) {
        for (int j = p.y - radius; j <= p.y + radius; ++j) {
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
    float slope;
    
    try {
        cv::solve (X_, Y_, sol, cv::DECOMP_SVD);
    }
    catch (...) {
        slope = 0;
    }
    
    slope = sol.at<float> (0);
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
    const cv::Point& point = cv::Point (p.first, p.second);
    
    const cv::Mat& confPatch = patch (point, _confidence);
    const int radius = (confPatch.rows - 1) / 2;
    const cv::Mat& pMask = patch (point, _mask, radius);
    
    cv::Mat maskedConfidence = cv::Mat::zeros (confPatch.size(),
                                               CV_64FC1);
    confPatch.copyTo (maskedConfidence, pMask == 0);
    
    const double confidence = (cv::sum (maskedConfidence)[0] /
                               confPatch.total()) * (1 - _w) +
                               _w;
    
//     const double confidence = (cv::sum (maskedConfidence)[0] /
//                                confPatch.total());
    
    const cv::Point2f& normal = generate_normal (point, radius);
    
    const cv::Mat& phi_p = patch (point, _modified, radius);
    
    cv::Mat gray;
    cv::cvtColor (phi_p, gray, cv::COLOR_RGB2GRAY);
    cv::Mat dx, dy, magnitude;
    cv::Sobel (gray, dx, CV_64F, 1, 0);
    cv::Sobel (gray, dy, CV_64F, 0, 1);
    
    cv::magnitude (dx, dy, magnitude);
    magnitude.setTo (0, pMask);
    
    cv::Mat erodedMagnitude;
    cv::erode (magnitude, erodedMagnitude, cv::Mat());
    
    cv::Point maxPoint;

    cv::minMaxLoc (erodedMagnitude, NULL, NULL, NULL, &maxPoint);
        
    const double dx_ = dx.at<double> (maxPoint);
    const double dy_ = dy.at<double> (maxPoint);
    
    const double dot = -dy_ * normal.x + dx_ * normal.y;
    
    return confidence * std::abs (dot);
}

/*****************************************************************************/ 

cv::Mat Criminisi::patch (const cv::Point& p, const cv::Mat& img)
{
    const int r[4] = {p.x, p.y, _cols - p.x, _rows - p.y};
    
    const int radius = std::min (*std::min_element (r, r + 4) - 1,
                                 _radius);
    
    return img  (cv::Range (p.y - radius, p.y + radius + 1),
                 cv::Range (p.x - radius, p.x + radius + 1));
}

/*****************************************************************************/

cv::Mat Criminisi::patch
(const cv::Point& p,const cv::Mat& img, const int radius)
{
    return img  (cv::Range (p.y - radius, p.y + radius + 1),
                 cv::Range (p.x - radius, p.x + radius + 1));
}

/*****************************************************************************/
/*****************************************************************************/