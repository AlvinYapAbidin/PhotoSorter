#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include "photosort.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    /* The goal of the project is to group pictures based on the subject, features and objects*/ 

    // Read each picture in the folder

    // Apply histogram to each picture, possibly add  feature detection for better accuracy

    // Use cluster algorithm to group similar 
    
    // Gui to help choose pictures within clusters



    //Photosort::run("/home/alvin/Documents/Projects/Computer_Vision_Projects/PhotoSorter/test");

    Photosort::run("photos");
}