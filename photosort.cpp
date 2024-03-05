#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ml.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include "utility.h"

// Didn't use namespace cv and std for learning purposes

namespace Photosort
{
    cv::Mat preprocess(cv::Mat image)
    {
        cv::resize(image, image, cv::Size(), 0.5, 0.5);

        cv::Mat imgGray;
        cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);

        cv::Mat imgBlurred;
        cv::GaussianBlur(imgGray, imgBlurred, cv::Size(3,3), 0);
        
        // cv::Mat imgSharpened;
        // cv::Laplacian(imgBlurred, imgSharpened, image.depth(), 3, 1, 0); 

        // cv::Mat imgContrast;
        // cv::createCLAHE()->apply(imgBlurred, imgContrast);
        
        // cv::Mat imgDenoised;
        // cv::fastNlMeansDenoisingColored(image, imgDenoised, 10, 10, 7, 21); // Another option for preprocessing

        return imgBlurred;
    }

    int run(std::string path)
    {
        std::string folder_path = path;
        std::vector<std::vector<cv::KeyPoint>> allKeypoints;
        std::vector<cv::Mat>  allDescriptors;
        std::vector<std::string> image_paths;
        
        if (!std::filesystem::exists(folder_path) || !std::filesystem::is_directory(folder_path)) 
        {
            std::cerr << "Directory does not exist or is not accessible: " << folder_path << std::endl;
            return -1;
        }
        
        std::cout << "Pick a detector: " << std::endl;
        std::cout << "Press 1 for ORB detector" << std::endl;
        std::cout << "Press 2 for SURF detector" << std::endl;
        std::cout << "Press 3 for SIFT detector" << std::endl;
        int userInput{};
        std::cin >> userInput;

        switch (userInput) 
        {
            case 1:
                std::cout << "You selected the ORB Detector." << std::endl;
                break;
            case 2:
                std::cout << "You selected the SIFT Detector." << std::endl;
                break;
            case 3:
                std::cout << "You selected the SURF Detector." << std::endl;
                break;
            default:
                std::cout << "Invalid choice. Please enter 1, 2, or 3." << std::endl;
        }

        std::cout << "Preprocessing and detecting keypoints using detector" << std::endl; // Update text to display appropriate detector.
        int totalFiles = Utility::countFilesInDirectory(folder_path);
        int processFiles{};

        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folder_path))
        {
            if (entry.is_regular_file())
            {
                std::string imagePath = entry.path().string();
                cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
                if (img.empty()) 
                {
                    std::cerr << "Failed to load image at " << imagePath << std::endl;
                    continue;
                }

                // Preprocess
                cv::Mat preprocessedImg = preprocess(img);

                std::vector<cv::KeyPoint> keyPoints{};
                cv::Mat descriptors{};

                if (userInput == 1) 
                {
                    cv::Ptr<cv::ORB> detector = cv::ORB::create();
                    std::vector<cv::KeyPoint> keyPoints;
                    cv::Mat descriptors;
                    detector->detectAndCompute(preprocessedImg, cv::noArray(), keyPoints, descriptors);
                    
                    allKeypoints.push_back(keyPoints);
                    allDescriptors.push_back(descriptors);
                    image_paths.push_back(imagePath);
                } 
                else if (userInput == 2) 
                {
                    int minHessian = 100;
                    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
                    std::vector<cv::KeyPoint> keyPoints;
                    cv::Mat descriptors;
                    detector->detectAndCompute(preprocessedImg, cv::noArray(), keyPoints, descriptors);
                    // FLANN matcher requires descriptors to be type 'CV_32F'
                    if (descriptors.type() != CV_32F) 
                    {
                        descriptors.convertTo(descriptors, CV_32F);
                    }

                    allKeypoints.push_back(keyPoints);
                    allDescriptors.push_back(descriptors);
                    image_paths.push_back(imagePath);
                } 
                else if (userInput == 3) 
                {
                    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
                    std::vector<cv::KeyPoint> keyPoints;
                    cv::Mat descriptors;
                    detector->detectAndCompute(preprocessedImg, cv::noArray(), keyPoints, descriptors);
                    // FLANN matcher requires descriptors to be type 'CV_32F'
                    if (descriptors.type() != CV_32F) 
                    {
                        descriptors.convertTo(descriptors, CV_32F);
                    }

                    allKeypoints.push_back(keyPoints);
                    allDescriptors.push_back(descriptors);
                    image_paths.push_back(imagePath);
                }


                Utility::updateProgressBar(++processFiles, totalFiles);
            }
        }
        std::cout << std::endl;

        // Cluster based on similarity score
        std::cout << "Clustering based on similarity" << std::endl;
        std::map<int, std::vector<int>> clusters{};

        if (userInput == 1)
        {
            const int MATCH_THRESHOLD = 20; 
            const float ratio_thresh = 0.75f;

            std::map<int, std::vector<int>> clusters = Clustering::clusterImagesBFMatcher(allDescriptors, MATCH_THRESHOLD, ratio_thresh);

            for (const std::pair<const int, std::vector<int>>& cluster : clusters) 
            {
                std::cout << "Cluster " << cluster.first + 1 << " has " << cluster.second.size() << " photos:\n";
                for (const int& index : cluster.second) 
                {
                    std::cout << " - " << image_paths[index] << std::endl; // Access path using index 
                }
            }

            std::cout << "Press any key to go to the next cluster." << std::endl;

            // Loop through each cluster and display its images in a grid
            for (const auto& cluster : clusters) {
                std::vector<std::string> clusterImagePaths;
                for (int index : cluster.second) {
                    clusterImagePaths.push_back(image_paths[index]); // Collect image paths for the current cluster
                }

                // Display images
                Utility::displayImagesGrid(clusterImagePaths, "Cluster " + std::to_string(cluster.first + 1));
            }
        }
        else if (userInput == 2) 
        {
            const int MATCH_THRESHOLD = 100; 
            const float ratio_thresh = 0.6f;

            std::map<int, std::vector<int>> clusters = Clustering::clusterImagesFLANN(allDescriptors, MATCH_THRESHOLD, ratio_thresh); // Use FLANN with SIFT and SURF dectector

            for (const std::pair<const int, std::vector<int>>& cluster : clusters) 
            {
                std::cout << "Cluster " << cluster.first + 1 << " has " << cluster.second.size() << " photos:\n";
                for (const int& index : cluster.second) 
                {
                    std::cout << " - " << image_paths[index] << std::endl; // Access path using index 
                }
            }

            std::cout << "Press any key to go to the next cluster." << std::endl;

            // Loop through each cluster and display its images in a grid
            for (const auto& cluster : clusters) {
                std::vector<std::string> clusterImagePaths;
                for (int index : cluster.second) {
                    clusterImagePaths.push_back(image_paths[index]); // Collect image paths for the current cluster
                }

                // Display images
                Utility::displayImagesGrid(clusterImagePaths, "Cluster " + std::to_string(cluster.first + 1));
            }
        }
        else if (userInput == 3)
        {
            const int MATCH_THRESHOLD = 80; 
            const float ratio_thresh = 0.6f;

            std::map<int, std::vector<int>> clusters = Clustering::clusterImagesFLANN(allDescriptors, MATCH_THRESHOLD, ratio_thresh); // Use FLANN with SIFT and SURF dectector

            for (const std::pair<const int, std::vector<int>>& cluster : clusters) 
            {
                std::cout << "Cluster " << cluster.first + 1 << " has " << cluster.second.size() << " photos:\n";
                for (const int& index : cluster.second) 
                {
                    std::cout << " - " << image_paths[index] << std::endl; // Access path using index 
                }
            }

            std::cout << "Press any key to go to the next cluster." << std::endl;

            // Loop through each cluster and display its images in a grid
            for (const auto& cluster : clusters) {
                std::vector<std::string> clusterImagePaths;
                for (int index : cluster.second) {
                    clusterImagePaths.push_back(image_paths[index]); // Collect image paths for the current cluster
                }

                // Display images
                Utility::displayImagesGrid(clusterImagePaths, "Cluster " + std::to_string(cluster.first + 1));
            }
        }
        return 0;
    }
}
    
#else
int main()
{
    std::cout << "The SURF  detector needs the xfeatures2d contriv module to be run" << std::endl;
    return 0;
}
#endif