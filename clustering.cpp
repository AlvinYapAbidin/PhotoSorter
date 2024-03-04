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
#include <string>
#include <iostream>
#include <map>

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
                matcher->knnMatch(allDescriptors[i], allDescriptors[cluster.second[0]], knn_matches, 2);
                
                // Filter matches using the Lowe's ratio test
                std::vector<cv::DMatch> good_matches;
                for (size_t j = 0; j < knn_matches.size(); j++)
                {
                    if (knn_matches[j][0].distance < ratioThresh * knn_matches[j][1].distance)
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
        }
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
        }
        return clusters;
    }
}