#ifndef SALIENCY_H
#define SALIENCY_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

using namespace cv;
using namespace std;

void SalientRegionDetectionBasedOnFT(unsigned char *Src, unsigned char* &SaliencyMap, int Width, int Height, int Stride);
void RGBToLABF(unsigned char* Src, float* dst, int width);

#endif