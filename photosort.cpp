#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <opencv2/ml.hpp>
#include <filesystem>
#include <vector>
#include <iostream>
#include <map>


namespace Photosort
{
    std::vector<cv::Mat> applyHistogram(const cv::Mat& img) // Assumed colour only
    {
        std::vector<cv::Mat> histogram;
        std::vector<cv::Mat> bgr_img;
        cv::split(img, bgr_img);

        int histSize = 256;
        float range[] = {0,256};
        const float* histRange = {range};

        for (int i = 0; i < bgr_img.size(); i++ )
        {
            cv::Mat hist;
            cv::calcHist(&bgr_img[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
            histogram.push_back(hist);
        }
        
        return histogram;
    }


    int run(std::string path)
    {
        std::string folder_path = path;
        std::vector<cv::Mat> feature_vectors;
        std::vector<std::string> image_paths;

        for (const auto& entry : std::filesystem::directory_iterator(folder_path))
        {
            if (entry.is_regular_file())
            {
                std::string imagePath = entry.path().string();
                cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Failed to load image at " << imagePath << std::endl;
                    continue; // Skip this image
                }

                auto histograms = applyHistogram(img);

                //Concatenate histogram into single feature vector
                cv::Mat featureVector;
                for (const auto& hist : histograms)
                {
                    featureVector.push_back(hist.reshape(1,1)); // Flatten and concatenate
                }

                feature_vectors.push_back(featureVector); // Store for clustering
                image_paths.push_back(imagePath); // Store image path
            }
        }

        cv::Mat samples;
        for (const auto& vec : feature_vectors)
        {
            samples.push_back(vec);
        }

        // Apply clustering elbow method

        // Conv to float
        samples.convertTo(samples, CV_32F);

        // Elbow method to find optimal K
        std::vector<double> wcss;
        int maxClusters = 10;
        for (int k = 1; k<= maxClusters; k++)
        {
            cv::Mat labels, centers;
            double compactness = cv::kmeans(samples, k, labels,
            cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 10, 1.0),
            3, cv::KMEANS_PP_CENTERS, centers);

            wcss.push_back(compactness);
            // Find the elbow point manually
            std::cout<< "WCSS for " << k << " clusters: " << compactness << std::endl;
        }

        int optimalK = 5;

        cv::Mat  labels, centers;
        
        cv::kmeans(samples, optimalK, labels,
            cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 10, 1.0),
            3, cv::KMEANS_PP_CENTERS, centers);
        

        //Organize photos by clusters
        std::map<int, std::vector<std::string>> clusteredPhotos;
        for (int i = 0; i < labels.rows; i++)
        {
            int clusterId = labels.at<int>(i, 0); // Make sure labels is CV_32S
            clusteredPhotos[clusterId].push_back(image_paths[i]);
        }

        // Print out or process clustered photos
        for (const auto& cluster : clusteredPhotos)
        {
            std::cout << "Cluster " << cluster.first << " has " << cluster.second.size() << " photos. " << std::endl;
            for (const auto& path : cluster.second)
            {
                std::cout << " - " << path << std::endl;
            }
        }

        // 2. Display clusters
        // 3. Add GUI


        return 0;
    }


    //// Alternative function that applies to colour and grayscale, do not really need grayscale
    // std::vector<cv::Mat> applyHistogram(const cv::Mat& img, bool isColour)
    // {
    //     std::vector<cv::Mat> histogram;

    //     if(isColour)
    //     {
    //         std::vector<cv::Mat> bgr_img;
    //         cv::split(img, bgr_img);

    //         int histSize = 256;
    //         float range[] = {0,256};
    //         const float* histRange = {range};

    //         for (int i = 0; i < bgr_img.size(); i++ )
    //         {
    //             cv::Mat hist;
    //             cv::calcHist(&bgr_img[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    //             histogram.push_back(hist);
    //         }
    //     }
    //     else
    //     {
    //         cv::Mat gray_img;
    //         cv::Mat hist;
            
    //         int histSize = 256;
    //         float range[] = {0,256};
    //         const float* histRange = {range};

    //         cv::calcHist(&gray_img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    //         histogram.push_back(hist);
    //     }
        
    //     return histogram;
    // }

}
