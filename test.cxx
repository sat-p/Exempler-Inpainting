#include "criminisi.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <cstdlib>

int main (int argc, char** argv)
{
    std::chrono::steady_clock clk;
    auto begin = clk.now();
    
    if (argc != 4) {
        std::cout << "Usage <INPUT_IMAGE_PATH> <MASK_HEIGHT_PIXELS> <MASK_WIDTH_PIXELS>"
                  << std::endl;
        return 0;
    }

    const auto& ref = cv::imread (argv[1]);

    cv::Mat lab_img;
    cv::cvtColor (ref, lab_img, cv::COLOR_BGR2Lab);
    
    const int N_ref = ref.rows * ref.cols;
    
    if (!N_ref) {
        
        std::cerr << "Unable to load the required image. "
                  << "Please enter correct path"
                  << std::endl;
                  
        return -1;
    }
    
    Criminisi criminisi (lab_img);
    
    const int x_size = std::atoi(argv[3]);
    const int y_size = std::atoi(argv[2]);
    
    cv::Mat mask = cv::Mat::zeros (ref.rows, ref.cols, CV_8UC1);
    cv::Mat roi = mask (cv::Rect ((ref.cols - x_size) / 2,
                                  (ref.rows - y_size) / 2,
                                  x_size,
                                  y_size));
    roi.setTo (255);
    
    criminisi.mask (mask);
    const auto& res_lab = criminisi.generate();
    cv::Mat res;
    
    cv::cvtColor (res_lab, res, cv::COLOR_Lab2BGR);

    cv::imshow ("ref", ref);
//     cv::imshow ("mask", mask);
    cv::imshow ("res", res);
    
    auto time = std::chrono::duration_cast<std::chrono::milliseconds> (clk.now() - begin);
    std::cerr << "time : " << time.count() << std::endl;
    
    std::cout << "Press 'q' to exit" << std::endl;
    
    while ('q' != cv::waitKey (1));
    
    return 0;
}