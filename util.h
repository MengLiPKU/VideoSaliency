#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <windows.h>
#include <io.h>
#include "Saliency.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

vector<string> getImagePath(string dirPath, bool subDir);
void drawMatch(Mat img1, Mat img2, ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matchInfo, Mat& drawImg);

#endif
