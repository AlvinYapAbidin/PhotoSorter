
#include <iostream>
#include "photosort.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    /* The goal of the project is to group pictures based on the subject, features and objects*/ 

    // 1. Read each picture in the folder

    // 2. Apply feature detection to get keypoints

    // 3. Use cluster algorithm to group similar 

    Photosort::run("test");

    // Photosort::run("photos");
}