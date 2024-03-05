#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ml.hpp>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <filesystem>
#include <string>
#include <iostream>
#include <map>

namespace Utility
{
    int countFilesInDirectory(const std::filesystem::path& path) 
    {
        return std::count_if(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator{},
                            [](const auto& entry) { return entry.is_regular_file(); });
    }

    void displayImagesGrid(const std::vector<std::string>& imagePaths, const std::string& windowName = "Cluster Images", int imagesPerRow = 5) 
    {
        if (imagePaths.empty()) {
            std::cout << "The cluster is empty." << std::endl;
            return;
        }

        // Determine grid size
        size_t numRows = (imagePaths.size() + imagesPerRow - 1) / imagesPerRow;
        int thumbWidth = 500; 
        int thumbHeight = 500; 

        // Create a large image to hold the grid
        cv::Mat gridImage(thumbHeight * numRows, thumbWidth * imagesPerRow, CV_8UC3, cv::Scalar(0, 0, 0));

        for (size_t i = 0; i < imagePaths.size(); ++i) {
            cv::Mat img = cv::imread(imagePaths[i]);
            if (img.empty()) {
                std::cerr << "Warning: Unable to read image: " << imagePaths[i] << std::endl;
                continue;
            }

            // Resize image to thumbnail size
            cv::Mat thumbnail;
            cv::resize(img, thumbnail, cv::Size(thumbWidth, thumbHeight));

            // Calculate position in the grid
            int row = i / imagesPerRow;
            int col = i % imagesPerRow;
            int startX = col * thumbWidth;
            int startY = row * thumbHeight;

            // Copy thumbnail into the correct grid position
            thumbnail.copyTo(gridImage(cv::Rect(startX, startY, thumbWidth, thumbHeight)));
        }

        // Display the grid
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, gridImage);
        cv::waitKey(0);
        cv::destroyWindow(windowName);
    }

    void updateProgressBar(int current, int total)
    {
        float progress = (current)/ (float)total;
        int barWidth  = 70;

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if ( i == pos) std::cout << "=";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }
}


namespace Clustering
{
    std::map<int, std::vector<int>> clusterImagesFLANN(const std::vector<cv::Mat>& allDescriptors, int matchThreshold, float ratioThresh)
    {
        std::map<int, std::vector<int>> clusters; // ClusterID to list of image indices        
        int clusterId = 0;

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

        for (int i = 0; i < allDescriptors.size(); i++)
        {
            bool foundCluster = false;
            for (std::pair<const int, std::vector<int>>& cluster : clusters)
            {

                // Here we match the current image descriptors with the first image of each cluster
                std::vector<std::vector<cv::DMatch>> knn_matches;
                if (!allDescriptors[i].empty() && allDescriptors[i].rows > 2 && !allDescriptors[cluster.second[0]].empty() && allDescriptors[cluster.second[0]].rows > 2)
                {
                    matcher->knnMatch(allDescriptors[i], allDescriptors[cluster.second[0]], knn_matches, 2);
                }
                
                // Filter matches using the Lowe's ratio test
                std::vector<cv::DMatch> good_matches;
                for (size_t j = 0; j < knn_matches.size(); j++)
                {
                    if (knn_matches[j].size() == 2 && knn_matches[j][0].distance < ratioThresh * knn_matches[j][1].distance)
                    {
                        good_matches.push_back(knn_matches[j][0]);
                    }
                }

                if (good_matches.size() > matchThreshold)
                {
                    // Add to cluster if current image is similar
                    cluster.second.push_back(i);
                    foundCluster = true;
                    break;
                }
                
            }
            if (!foundCluster) 
            {
                clusters[clusterId++] = std::vector<int>{i}; // A new cluster is created for the image if no similar is found
            }

            Utility::updateProgressBar(i, allDescriptors.size());
        }
        std::cout << std::endl;
        return clusters;
    }

    std::map<int, std::vector<int>> clusterImagesBFMatcher(const std::vector<cv::Mat>& allDescriptors, int matchThreshold, float ratioThresh)
    {
        std::map<int, std::vector<int>> clusters; // ClusterID to list of image indices        
        int clusterId = 0;

        cv::BFMatcher matcher(cv::NORM_HAMMING); // We use NORM_HAMMING because ORB currently set to WTA_K = 2

        for (int i = 0; i < allDescriptors.size(); i++)
        {
            bool foundCluster = false;
            for (std::pair<const int, std::vector<int>>& cluster : clusters)
            {
                // Here we match the current image descriptors with the first image of each cluster
                std::vector<std::vector<cv::DMatch>> knn_matches;
                matcher.knnMatch(allDescriptors[i], allDescriptors[cluster.second[0]], knn_matches, 2);
                
                // Filter matches using the Lowe's ratio test
                std::vector<cv::DMatch> good_matches;
                for (size_t j = 0; j < knn_matches.size(); j++)
                {
                    if (knn_matches[j].size() == 2 && knn_matches[j][0].distance < ratioThresh * knn_matches[j][1].distance)
                    {
                        good_matches.push_back(knn_matches[j][0]);
                    }
                }

                if (good_matches.size() > matchThreshold)
                {
                    // Add to cluster if current image is similar
                    cluster.second.push_back(i);
                    foundCluster = true;
                    break;
                }
                
            }
            if (!foundCluster) 
            {
                clusters[clusterId++] = std::vector<int>{i}; // A new cluster is created for the image if no similar is found
            }
            Utility::updateProgressBar(i, allDescriptors.size());
            
        }
        std::cout << std::endl;
        return clusters;
    }
}
