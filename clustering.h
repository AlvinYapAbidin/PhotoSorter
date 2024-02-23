#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <map>

namespace Clustering
{
    std::map<int, std::vector<int>> clusterImagesFLANN(const std::vector<cv::Mat>& allDescriptors, int matchThreshold, float ratioThresh);
    std::map<int, std::vector<int>> clusterImagesBFMatcher(const std::vector<cv::Mat>& allDescriptors, int matchThreshold, float ratioThresh);

}

#endif