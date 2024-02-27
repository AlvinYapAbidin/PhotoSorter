# PhotoSorter

## Overview

PhotoSort is an application designed to help users automatically organize their photo collections into clusters based on visual similarity. The application leverages OpenCV's powerful image processing and feature detection algorithms to analyze and group images.
Prerequisites

- C++17 compatible compiler (GCC, Clang, MSVC, etc.)
- CMake (version 3.14 or higher)
- OpenCV (version 4.x)
- A basic understanding of terminal or command prompt usage

## Installation
Step 1: Install OpenCV

You need to have OpenCV installed on your system. You can download and install OpenCV from the official OpenCV website or use a package manager like vcpkg, brew, or apt-get, depending on your operating system.

For example, on Ubuntu, you can install OpenCV using the following command:

    sudo apt-get install libopencv-dev

Step 2: Compile the Application

1. Clone or download the application source code to your local machine.
2. Navigate to the application's root directory.
3. Create a new directory for the build process and navigate into it:

        mkdir build && cd build

4. Run CMake to configure the project and generate makefiles:


        cmake ..

5. Compile the project:


        make

## Demonstration
![Alt Text](https://github.com/AlvinYapAbidin/PhotoSorter/blob/main/demorun.gif)

## Usage

To run the application, you need to provide the path to the directory containing your images as an argument to the executable. From the build directory, execute the following command:

    ./PhotoSort /path/to/your/images

The application will process the images, detecting keypoints and descriptors, and then cluster the images based on similarity. Each cluster's images will be displayed in a window.

The provided code comprises two parts, photosort.cpp and clustering.cpp, which are designed to sort and cluster images based on their visual similarity. Here's a breakdown of what each part does and some suggestions for improvement or considerations in case you encounter any issues.

### photosort.cpp Overview

- Functionality: This part of the code handles reading images from a directory, preprocessing them, detecting keypoints using the ORB detector (originally designed for SIFT but changed to ORB), and then clustering the images based on the similarity of their features. It also includes functions for displaying images in a grid format, which is useful for visualizing clusters.

- Preprocessing: Images are resized and converted to grayscale before applying a Gaussian blur and Laplacian filter for sharpening. There's also commented code for denoising and contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).

- Key Points Detection: Utilizes the ORB detector to find key points and compute descriptors for each image.

- Clustering: The run function manages the workflow, including clustering which is performed in the clustering.cpp part of the code.

- Progress Bar: A simple progress bar is implemented to give feedback during the processing of images.

### clustering.cpp Overview

- Clustering Functionality: Contains two functions for clustering images based on the similarity of their features. One uses FLANN for approximate nearest neighbors search, suitable for SIFT descriptors, and the other uses BFMatcher, which is used with the ORB descriptors.

- FLANN vs. BFMatcher: The choice between FLANN and BFMatcher depends on the type of descriptors being used and the specific requirements of the application. FLANN is generally faster for large datasets but may be less accurate than BFMatcher.

### Suggestions and Considerations

- ORB vs. SIFT: The change from SIFT to ORB might significantly affect the quality and performance of the feature matching process. ORB is faster and less memory-intensive than SIFT but might be less accurate in some cases. It is essential to consider the trade-offs based on your application's needs.

- Error Handling: Make sure to handle errors gracefully, especially when reading images or processing directories that might not exist or be accessible.

- Performance Optimizations: Depending on the size of your dataset, processing can be quite slow. Consider parallelizing the processing of images or using down-sampled images for the initial clustering phase to improve performance.

- Parameter Tuning: The effectiveness of clustering heavily depends on the parameters used (e.g., MATCH_THRESHOLD, ratio_thresh). You might need to experiment with these values to find the best results for your specific dataset.

- Visualizing Clusters: The displayImagesGrid function is used to visualize the results. However, for large datasets, consider additional methods to explore and analyze the clusters, such as saving cluster information to a file or database.

- Dependencies: Ensure that all dependencies, especially OpenCV and its contrib modules, are correctly installed and compatible with the version of the code.

# Contributing

I welcome contributions to the PhotoSorter project! If you have suggestions for improvements or bug fixes, please feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Contact

For questions and support, please contact email alvinyapabidin@gmail.com

