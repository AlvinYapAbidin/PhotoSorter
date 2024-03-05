#ifndef UTILITY_H
#define UTILITY_H

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <iostream>
#include <map>

namespace Clustering
{
    std::map<int, std::vector<int>> clusterImagesFLANN(const std::vector<cv::Mat>& allDescriptors, int matchThreshold, float ratioThresh);
    std::map<int, std::vector<int>> clusterImagesBFMatcher(const std::vector<cv::Mat>& allDescriptors, int matchThreshold, float ratioThresh);

}

namespace Utility
{
    int countFilesInDirectory(const std::filesystem::path& path);
    void displayImagesGrid(const std::vector<std::string>& imagePaths, const std::string& windowName = "Cluster Images", int imagesPerRow = 5);
    void updateProgressBar(int current, int total);

}


#endif