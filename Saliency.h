#ifndef SALIENCY_H
#define SALIENCY_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

using namespace cv;
using namespace std;

void saliencyDetect(vector<string> imgPath);
Mat spatialSaliency(unsigned char *Src, int Width, int Height, int Stride);
Mat temporalSaliency(Mat img1, Mat img2);
Mat blurSpatialTemporal(Mat sSaliency, Mat tSaliency);

#endif