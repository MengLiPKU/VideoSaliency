#include "util.h"

vector<string> getImagePath(string dirPath, bool subDir) {
	vector<string> ans; ans.clear();

	string dir_spec = dirPath + "/*.jpg";
	WIN32_FIND_DATAA f;
	HANDLE h = FindFirstFileA(dir_spec.c_str(), &f);
	if(h != INVALID_HANDLE_VALUE) {
		do {
			ans.push_back(dirPath + "/" + string(f.cFileName));
		}while(FindNextFileA(h, &f));
	}
	FindClose(h);

	dir_spec = dirPath + "/*.png";
	h = FindFirstFileA(dir_spec.c_str(), &f);
	if(h != INVALID_HANDLE_VALUE) {
		do {
			ans.push_back(dirPath + "/" + string(f.cFileName));
		}while(FindNextFileA(h, &f));
	}
	FindClose(h);

	return ans;
}

void drawMatch(Mat img1, Mat img2, ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matchInfo, Mat& drawImg) {
	vector<Point2f> srcPts, dstPts;
	for(int i = 0; i < matchInfo.matches.size(); i++) {
		const DMatch& m = matchInfo.matches[i];
		Point2f p1 = feat1.keypoints[m.queryIdx].pt;
		Point2f p2 = feat2.keypoints[m.trainIdx].pt;
		if(matchInfo.inliers_mask[i]) {
			srcPts.push_back(p1);
			dstPts.push_back(p2);
		}
	}

	int row1 = img1.rows;
	int row2 = img2.rows;
	int col1 = img1.cols;
	int col2 = img2.cols;
	drawImg.create(max(row1, row2), col1 + col2, CV_8UC3);
	img1.copyTo(drawImg(Rect(0, 0, col1, row1)));
	img2.copyTo(drawImg(Rect(col1, 0, col2, row2)));
	int numPts = srcPts.size();
	int radius = 2;
	Scalar colorPt = Scalar(255, 0, 0, 0);
	Scalar colorLine = Scalar(0, 0, 255, 0);

	for(int i = 0; i < numPts; i++) {
		circle(drawImg, Point(srcPts[i]), radius, colorPt, radius);
		circle(drawImg, Point(dstPts[i])+Point(col1, 0), radius, colorPt, radius);
		line(drawImg, Point(srcPts[i]), Point(dstPts[i])+Point(col1, 0), colorLine);
	}
}

