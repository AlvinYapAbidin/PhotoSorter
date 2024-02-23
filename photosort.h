#ifndef PHOTOSORT_H
#define PHOTOSORT_H

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;

namespace Photosort
{
    int run(std::string path);
}

#endif