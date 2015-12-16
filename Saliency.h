#ifndef SALIENCY_H
#define SALIENCY_H

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

class homography {
public:
	vector<Point2f> inliers;
	vector<Point2f> corresponding;
	int size;
	Mat H;
	int maxX, maxY, minX, minY;
	double aveSaliency;

	homography() {
		inliers.clear();
		maxX = maxY = minX = minY = 0;
	}
};

void saliencyDetect(vector<string> imgPath);
Mat spatialSaliency(unsigned char *Src, int Width, int Height, int Stride);
Mat temporalSaliency(Mat img1, Mat img2);
Mat blurSpatialTemporal(Mat sSaliency, Mat tSaliency);

#endif