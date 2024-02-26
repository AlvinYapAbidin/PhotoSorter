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
#include "clustering.h"

// Didn't use namespace cv and std for learning purposes


namespace Photosort
{
    int countFilesInDirectory(const std::filesystem::path& path) {
        return std::count_if(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator{},
                            [](const auto& entry) { return entry.is_regular_file(); });
    }

    void displayImagesGrid(const std::vector<std::string>& imagePaths, const std::string& windowName = "Cluster Images", int imagesPerRow = 5) {
        if (imagePaths.empty()) {
            std::cout << "The cluster is empty." << std::endl;
            return;
        }

        // Determine grid size
        size_t numRows = (imagePaths.size() + imagesPerRow - 1) / imagesPerRow;
        int thumbWidth = 500; // Width of each thumbnail
        int thumbHeight = 500; // Height of each thumbnail

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
        cv::waitKey(0); // Wait for any key press to close
        cv::destroyWindow(windowName);
    }

    cv::Mat preprocess(cv::Mat image)
    {
        cv::resize(image, image, cv::Size(), 0.5, 0.5);

        cv::Mat imgGray;
        cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);

        cv::Mat imgBlurred;
        cv::GaussianBlur(imgGray, imgBlurred, cv::Size(3,3), 0);
        
        cv::Mat imgSharpened;
        Laplacian(imgBlurred, imgSharpened, image.depth(), 3, 1, 0); 

        cv::Mat imgContrast;
        cv::createCLAHE()->apply(imgBlurred, imgContrast);
        
        //cv::Mat imgDenoised;
        //cv::fastNlMeansDenoisingColored(image, imgDenoised, 10, 10, 7, 21);

        return imgContrast;
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
        
        std::cout << "Preprocessing and detecting keypoints using SIFT detector" << std::endl;
        int totalFiles = countFilesInDirectory(folder_path);
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

                // cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
                cv::Ptr<cv::ORB> detector = cv::ORB::create();
                std::vector<cv::KeyPoint> keyPoints;
                cv::Mat descriptors;

                detector->detectAndCompute(preprocessedImg, cv::noArray(), keyPoints, descriptors);

                // // FLANN matcher requires descriptors to be type 'CV_32F'
                // if (descriptors.type() != CV_32F) 
                // {
                //     descriptors.convertTo(descriptors, CV_32F);
                // }

                allKeypoints.push_back(keyPoints);
                allDescriptors.push_back(descriptors);
                image_paths.push_back(imagePath);
                
                updateProgressBar(++processFiles, totalFiles);
            }
        }
        std::cout << std::endl;


        // Cluster based on similarity score
        std::cout << "Clustering based on similarity" << std::endl;

        const int MATCH_THRESHOLD = 20; // Minimum number of matches required
        const float ratio_thresh = 0.75f; // Lowe's ratio threshold for filtering matches

        //std::map<int, std::vector<int>> clusters = Clustering::clusterImagesFLANN(allDescriptors, MATCH_THRESHOLD, ratio_thresh); // Use FLANN with SIFT dectector

        std::map<int, std::vector<int>> clusters = Clustering::clusterImagesBFMatcher(allDescriptors, MATCH_THRESHOLD, ratio_thresh); // Use BFMatcher with ORB detector

        std::cout << std::endl;
    

        // Printing each of the clusters
        for (const std::pair<const int, std::vector<int>>& cluster : clusters) 
        {
            std::cout << "Cluster " << cluster.first << " has " << cluster.second.size() << " photos:\n";
            for (const int& index : cluster.second) 
            {
                std::cout << " - " << image_paths[index] << std::endl; // Access path using index
            }
        }

        // Loop through each cluster and display its images in a grid
        for (const auto& cluster : clusters) {
            std::vector<std::string> clusterImagePaths;
            for (int index : cluster.second) {
                clusterImagePaths.push_back(image_paths[index]); // Collect image paths for the current cluster
            }

            // Display images in a grid
            displayImagesGrid(clusterImagePaths, "Cluster " + std::to_string(cluster.first));
        }



        return 0;
    }
}
    
