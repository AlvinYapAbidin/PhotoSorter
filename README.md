# PhotoSorter

## Capturing the perfect moment

In the quest for the that perfect shot, photographers like myself often find themselves capturing multiple photos of the same scene to  ensure they're got the best possible selection. The challenge, however, arises during the editing process: sifting through these batches to unearth the gems can be a daunting task. PhotoSort is a solution designed looks to change the way photos are organised by automatically clustering images based on their visual similarities.

## Demonstration
![Alt Text](https://github.com/AlvinYapAbidin/PhotoSorter/blob/main/demorun.gif)


## Key Features
- Automatic Clustering: Streamline the organization of your photo collections by grouping images that share visual similarities.
- Advanced Image Processing: Utilizes OpenCV's robust image processing capabilities to enhance feature matching.
- Feature Matching with SURF: Incorporates the SURF (Speeded Up Robust Features) detector for improved detection of keypoints and descriptors, offering a significant boost in the quality of clustering compared to ORB and SIFT.
- NOTE: SURF detector requires Opencv to be built with the non-free version as the SURF detector is patented

## Installation

### Prerequisites

Before installing PhotoSorter, ensure that OpenCV is installed on your system. This library has been tested with OpenCV version 4.0 and later, including the non-free modules for advanced feature detection with SURF. For detailed installation instructions of OpenCV, please refer to the official OpenCV documentation. A basic understanding of terminal or command prompt usage is required.

### Building the Library

To compile the Vision Library, follow these steps:

1. Clone the repository to your local machine.

       git clone https://github.com/AlvinYapAbidin/PhotoSorter.git
       cd PhotoSorter

2. Use CMake to prepare the build environment. 

       cmake .

3. Compile the source code.

       make

## Usage

1. To specify the folder of pictures, you must enter the folder name within the main function in the main.cpp file:

       Photosort::run("path/to/folder");

2. Build the project using CMake as detailed in the installation section.

3. To run the application,enter the program executable specifed in the CMakeLists.txt file, within the terminal (ensure that the files have already been built with CMake after editing):

       ./photosort

The application will process the images, detecting keypoints and descriptors, and then cluster the images based on similarity. Each cluster's images will be displayed in a window.

The provided code comprises two parts, photosort.cpp and clustering.cpp, which are designed to sort and cluster images based on their visual similarity. Here's a breakdown of what each part does and some suggestions for improvement or considerations in case you encounter any issues.

### photosort.cpp Overview

- Functionality: This part of the code handles reading images from a directory, preprocessing them, detecting keypoints using the ORB detector (originally designed for SIFT but changed to ORB), and then clustering the images based on the similarity of their features. It also includes functions for displaying images in a grid format, which is useful for visualizing clusters.

- Preprocessing: Images are resized and converted to grayscale before applying a Gaussian blur and Laplacian filter for sharpening. There's also commented code for denoising and contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).

- Key Points Detection: Utilizes the SURF detector to find key points and compute descriptors for each image. If on the free version, you can try the ORB detector or SIFT detector

- Clustering: The run function manages the workflow, including clustering which is performed in the clustering.cpp part of the code.

- Progress Bar: A simple progress bar is implemented to give feedback during the processing of images.

### clustering.cpp Overview

- Clustering Functionality: Contains two functions for clustering images based on the similarity of their features. One uses FLANN for approximate nearest neighbors search, suitable for SURF and SIFT descriptors, and the other uses BFMatcher, which is used with the ORB descriptors.

- FLANN vs. BFMatcher: The choice between FLANN and BFMatcher depends on the type of descriptors being used and the specific requirements of the application. FLANN is generally faster for large datasets but may be less accurate than BFMatcher.

### Personal Findings and Considerations

- SURF vs ORB vs. SIFT - For image clustering, where the goal is to group images based on visual similarities without prior labels:

    SIFT might be preferred for datasets where accuracy in capturing nuanced distinctions between images is paramount, and computational resources are not a major constraint.
    SURF could offer a good balance if your application requires somewhat faster processing while still maintaining robustness to scale and rotation.
    ORB might be the best choice for large-scale applications or when working under tight computational constraints, as long as the reduced scale invariance does not significantly impact the quality of the clustering.

- Error Handling: Make sure to handle errors gracefully, especially when reading images or processing directories that might not exist or be accessible.

- Performance Optimizations: Depending on the size of your dataset, processing can be quite slow. Consider parallelizing the processing of images or using down-sampled images for the initial clustering phase to improve performance.

- Parameter Tuning: The effectiveness of clustering heavily depends on the parameters used (e.g., MATCH_THRESHOLD, ratio_thresh). You might need to experiment with these values to find the best results for your specific dataset.

- Visualizing Clusters: The displayImagesGrid function is used to visualize the results. However, for large datasets, consider additional methods to explore and analyze the clusters, such as saving cluster information to a file or database.

- Dependencies: Ensure that all dependencies, especially OpenCV and its contrib modules, are correctly installed and compatible with the version of the code.

- Clasical approach requires manual testing for Match number threshold and Lowe's ration threshold

# Contributing

I welcome contributions to the PhotoSorter project! If you have suggestions for improvements or bug fixes, please feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Contact

For questions and support, please contact email alvinyapabidin@gmail.com

