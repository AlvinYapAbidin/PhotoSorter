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

// Didn't use namespace cv and std for learning purposes
using namespace cv;
using namespace cv::xfeatures2d;


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
        int thumbWidth = 300; // Width of each thumbnail
        int thumbHeight = 300; // Height of each thumbnail

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

        cv::Mat imgBlurred;
        cv::GaussianBlur(image, imgBlurred, cv::Size(3,3), 0);

        // cv::Mat imgSharpened;
        // Laplacian(imgBlurred, imgSharpened, image.depth(), 3, 1, 0); 
        
        //cv::Mat imgDenoised;
        //cv::fastNlMeansDenoisingColored(image, imgDenoised, 10, 10, 7, 21);

        return imgBlurred;
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

                // // Preprocessing
                // cv::Mat resizedImg;
                // cv::resize(img, resizedImg, cv::Size(), 0.1, 0.1); // Resized to make program run faster

                // Preprocess
                cv::Mat preprocessedImg = preprocess(img);

                // Detect all of the keypoints
                int minHessian = 400;
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
                
                updateProgressBar(++processFiles, totalFiles);
            }
        }
        std::cout << std::endl;



        // Cluster based on similarity score
        std::cout << "Clustering based on similarity" << std::endl;
        processFiles = 0;

        std::map<int, std::vector<int>> clusters; // ClusterID to list of image indices        
        int clusterId = 0;
        const int MATCH_THRESHOLD = 80; // Minimum number of matches required
        const float ratio_thresh = 0.6f; // Lowe's ratio threshold for filtering matches

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

        for (int i = 0; i < allDescriptors.size(); i++)
        {
            bool foundCluster = false;
            for (std::pair<const int, std::vector<int>>& cluster : clusters)
            {
                // Here we match the current image descriptors with the first image of each cluster
                std::vector<std::vector<cv::DMatch>> knn_matches;
                matcher->cv::DescriptorMatcher::knnMatch(allDescriptors[i], allDescriptors[cluster.second[0]], knn_matches, 2);
                
                // Filter matches using the Lowe's ratio test
                std::vector<DMatch> good_matches;
                for (size_t j = 0; j < knn_matches.size(); j++)
                {
                    if (knn_matches[j].size() == 2 && knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)
                    {
                        good_matches.push_back(knn_matches[j][0]);
                    }
                }

                if (good_matches.size() > MATCH_THRESHOLD)
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
            updateProgressBar(++processFiles, totalFiles);
        }

        std::cout << std::endl;
    
        // Use the indices to refer back to the image paths
        for (const std::pair<const int, std::vector<int>>& cluster : clusters) 
        {
            std::cout << "Cluster " << cluster.first << " has " << cluster.second.size() << " photos:\n";
            for (const int& index : cluster.second) 
            {
                std::cout << " - " << image_paths[index] << std::endl; // Access path using index
            }
        }


        // // Displaying the images
        // int windowWidth = 800;
        // int windowHeight = 600; 
        
        // for (const std::pair<const int, std::vector<int>>& cluster : clusters) {
        //     std::string windowName = "Cluster " + std::to_string(cluster.first);
        //     // cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        //     // cv::resizeWindow(windowName, windowWidth, windowHeight);

        //     for (const int& index : cluster.second) {
        //         cv::Mat img = cv::imread(image_paths[index]); 
        //         if (!img.empty()) {
        //             cv::Mat resizedImg;
        //             float aspectRatio = (float)img.cols / (float)img.rows;
        //             int resizedWidth, resizedHeight;
        //             if (aspectRatio > 1) { // Image is wider than it is tall
        //                 resizedWidth = windowWidth;
        //                 resizedHeight = static_cast<int>(windowWidth / aspectRatio);
        //             } else { // Image is taller than it is wide or square
        //                 resizedHeight = windowHeight;
        //                 resizedWidth = static_cast<int>(windowHeight * aspectRatio);
        //             }
        //             cv::resize(img, resizedImg, cv::Size(resizedWidth, resizedHeight));

        //             // Display image
        //             cv::imshow(windowName, resizedImg); // Display the resized image
        //             int key = cv::waitKey(0); // Wait for a key press to move to the next image
        //             if (key == 27) break; // Optional: break on ESC key to move to the next cluster
        //         }
        //     }
        //     cv::destroyWindow(windowName);
        // }

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
    
