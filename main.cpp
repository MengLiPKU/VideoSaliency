#include "Saliency.h"

int main() {
	std::string inputPath;
	std::cin >> inputPath;
	Mat img = imread(inputPath);
	unsigned char* saliencyData = NULL;
	SalientRegionDetectionBasedOnFT(img.data, saliencyData, img.cols, img.rows, img.cols * img.channels());
	Mat saliencyImg = Mat(img.rows, img.cols, CV_8UC1, saliencyData);
	imshow("test", saliencyImg);
	cvWaitKey(0);
}