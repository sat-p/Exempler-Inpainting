#include "criminisi.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main (int argc, char** argv)
{
    if (argc != 2) {
    
        std::cout << "Please input reference image path" << std::endl;
        return 0;
    }

    const auto& ref = cv::imread (argv[1]);

    const int N_ref = ref.rows * ref.cols;
    
    if (!N_ref) {
        
        std::cerr << "Unable to load the required image. "
                  << "Please enter correct path"
                  << std::endl;
                  
        return -1;
    }
    
    Criminisi criminisi (ref);
    
    const int x_size = ref.cols / 3;
    const int y_size = ref.rows / 3;
    
    cv::Mat mask = cv::Mat::zeros (ref.rows, ref.cols, CV_8UC1);
    cv::Mat roi = mask (cv::Rect ((ref.cols - x_size) / 2,
                                  (ref.rows - y_size) / 2,
                                  x_size,
                                  y_size));
    roi.setTo (255);
    
    criminisi.mask (mask);
    const auto& res = criminisi.generate();
    
    cv::imshow ("ref", ref);
    cv::imshow ("mask", mask);
//     cv::imshow ("res", res);
    
    cv::Mat mask2 = cv::Mat::zeros (ref.rows, ref.cols, CV_8UC1);
    
    criminisi.draw_contour (mask2, 255);
    cv::imshow ("contour", mask2);
    
    std::cout << "Press 'q' to exit" << std::endl;
    
    while ('q' != cv::waitKey (1));
    
    return 0;
}