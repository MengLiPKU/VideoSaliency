#include "Saliency.h"
#include "util.h"
#include <limits.h>
#include <math.h>

using namespace cv::detail;

#define RANSACTHRESHOLD 15
#define DISTANCERATIO 10
#define SIZERATIOTHRES 0.65

string curDir;
void RGBtoLAB(unsigned char* Src, vector<double>& lvec, vector<double>& avec, vector<double>& bvec, int width) {
	for(int i = 0; i < width; i++) {
		int sR = (int)(Src[3*i+2]);
		int sG = (int)(Src[3*i+1]);
		int sB = (int)(Src[3*i]);
		double R = sR / 255.0;
		double G = sG / 255.0;
		double B = sB / 255.0;
		double r, g, b;
		
		if(R <= 0.04045)	r = R/12.92;
		else				r = pow((R+0.055)/1.055,2.4);
		if(G <= 0.04045)	g = G/12.92;
		else				g = pow((G+0.055)/1.055,2.4);
		if(B <= 0.04045)	b = B/12.92;
		else				b = pow((B+0.055)/1.055,2.4);

		double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
		double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
		double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
		
		double epsilon = 0.008856;	//actual CIE standard
		double kappa   = 903.3;		//actual CIE standard

		double Xr = 0.950456;	//reference white
		double Yr = 1.0;		//reference white
		double Zr = 1.088754;	//reference white

		double xr = X/Xr;
		double yr = Y/Yr;
		double zr = Z/Zr;

		double fx, fy, fz;
		if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
		else				fx = (kappa*xr + 16.0)/116.0;
		if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
		else				fy = (kappa*yr + 16.0)/116.0;
		if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
		else				fz = (kappa*zr + 16.0)/116.0;

		lvec.push_back(116.0 * fy - 16.0);
		avec.push_back(500.0 * (fx - fy));
		bvec.push_back(200.0 * (fy - fz));
	}
}

void GuassBlur(vector<double> inputImage, int width, int height, vector<double>& smoothImg) {
	vector<double> kernel(0);
	kernel.push_back(1.0);
	kernel.push_back(2.0);
	kernel.push_back(1.0);
	
	int center = int(kernel.size()) / 2;
	int size = width * height;
	smoothImg.clear();
	smoothImg.resize(size);
	vector<double> tempim(size);
	int rows = height;
	int cols = width;

	int index = 0;
	for(int r = 0;  r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			double kernelSum(0);
			double sum(0);
			for(int cc = (-center); cc <= center; cc++) {
				if(((c + cc) >= 0) && ((c + cc) < cols)) {
					sum += inputImage[r*cols + (c + cc)] * kernel[center + cc];
					kernelSum += kernel[center + cc];
				}
			}
			tempim[index] += sum / kernelSum;
			index++;
		}
	}

	index = 0;
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			double kernelSum(0);
			double sum(0);
			for(int rr = (-center); rr <= center; rr++) {
				if(((r + rr) >= 0) && ((r + rr) < rows)) {
					sum += tempim[(r+rr) * cols + c] * kernel[center + rr];
					kernelSum += kernel[center + rr];
				}
			}
			smoothImg[index] = sum / kernelSum;
			index++;
		}
	}
}

void Normalize(vector<double> salMap, vector<double>& normaledMap, int width, int height) {
	double maxVal = 0;
	double minVal = DBL_MAX;
	int i = 0;
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			maxVal = max(maxVal, salMap[i]);
			minVal = min(minVal, salMap[i]);
			i++;
		}
	}
	double range = maxVal - minVal;
	if(range == 0)
		range = 1;
	i = 0;
	normaledMap.clear();
	normaledMap.resize(width * height);
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			normaledMap[i] = 255 * (salMap[i]-minVal) / range;
			i++;
		}
	}
}

void vec2uchar(vector<double> normaledSalmap, unsigned char* &saliencyMap) {
	saliencyMap = new unsigned char[normaledSalmap.size()];
	for(int i = 0;  i < normaledSalmap.size(); i++) {
		saliencyMap[i] = (unsigned char)((int)normaledSalmap[i]);
	}
}

Mat spatialSaliency(unsigned char *src, int width, int height, int stride)
{
    double meanL = 0, meanA = 0, meanB = 0;
    
	vector<double> lvec(0);
	vector<double> avec(0);
	vector<double> bvec(0);

	for (int y = 0; y < height ; y++) 
        RGBtoLAB(src + y * stride, lvec, avec, bvec, width);                //    浮点类型的数据转换
    
	for (int y = 0; y < height; y++) 
    {
        int index = y * width;
        for (int x = 0; x < width; x++)
        {
            meanL +=  lvec[index];
            meanA +=  avec[index];
            meanB +=  bvec[index];
            index++;
        }
    }
	meanL /= (width * height);                                            //    求LAB空间的平均值
    meanA /= (width * height);
    meanB /= (width * height);
	vector<double> slvec(0);
	vector<double> savec(0);
	vector<double> sbvec(0);
    GuassBlur(lvec, width, height, slvec);                            //    use Gaussian blur to eliminate ﬁne texture details as well as noise and coding artifacts
	GuassBlur(avec, width, height, savec); 
	GuassBlur(bvec, width, height, sbvec);
	vector<double> salmap;
	vector<double> normaledSalmap;
	salmap.clear(); normaledSalmap.clear();
	salmap.resize(width * height); normaledSalmap.resize(width * height);

	for(int i = 0; i < width * height; i++) {
		salmap[i] = (meanL - slvec[i]) *  (meanL - slvec[i]) +  (meanA - savec[i]) *  (meanA - savec[i]) +  (meanB - sbvec[i]) *  (meanB - sbvec[i]);
    }
	Normalize(salmap, normaledSalmap, width, height);                //    归一化图像数据

	unsigned char* saliencyMap = new unsigned char[width * height];
	vec2uchar(normaledSalmap, saliencyMap);

	Mat ans = Mat(height, width, CV_8UC1, saliencyMap);
	return ans;
}

void calDistance(vector<homography>& hvec, int intimageSize) {
	double imageSize = double(intimageSize);
	double maxSal = 0;
	for(int i = 0; i < hvec.size(); i++) {     //每个单应,一次计算所有点的透视变换
		double distance = 0; 
		for(int j = 0; j < hvec.size(); j++) {        //每个单应
			vector<Point2f> perspected;
			perspectiveTransform(hvec[i].inliers, perspected, hvec[j].H);
			for(int k = 0; k < perspected.size(); k++) {            //所有点距离求和
				double tmpDistance = (hvec[i].corresponding[k].x - perspected[k].x) * (hvec[i].corresponding[k].x - perspected[k].x) + (hvec[i].corresponding[k].y - perspected[k].y)*(hvec[i].corresponding[k].y - perspected[k].y);
				distance += sqrt(tmpDistance) * hvec[j].size / imageSize;
				//printf("(%f, %f), (%f, %f)\n", hvec[i].corresponding[k].x, hvec[i].corresponding[k].y, perspected[k].x, perspected[k].y);
			}
		}
		hvec[i].aveSaliency = distance / (1.0 * hvec[i].inliers.size() * hvec[i].size);
		maxSal = max(maxSal, hvec[i].aveSaliency);
	}
	
	if(hvec.size() == 1 && hvec[0].size / (1.0*intimageSize) > SIZERATIOTHRES) {
		hvec[0].aveSaliency = 0;
		return;
	}
	for(int i = 0; i < hvec.size(); i++) {
		hvec[i].aveSaliency = 128 * (hvec[i].aveSaliency / maxSal);
	}
}

int imageID = 0;
Mat getTemporalSaliency(Mat img, vector<Point2f> pointsPre, vector<Point2f> pointsCur) {
	Mat tSaliency = Mat::zeros(img.rows, img.cols, CV_8UC1);
	
	vector<homography> hvec;
	while(pointsPre.size() > 4) {
		homography curHomography;
		Mat mask;
		Mat H = findHomography(pointsPre, pointsCur, CV_RANSAC, RANSACTHRESHOLD, mask);
		uchar* p = mask.ptr<uchar>(0);
		int maxX = 0, maxY = 0, minX = INT_MAX, minY = INT_MAX;
		for(int i = mask.rows - 1; i >= 0; i--) {
			if((int)p[i]) {
				maxX = max(maxX, pointsPre[i].x);
				maxY = max(maxY, pointsPre[i].y);
				minX = min(minX, pointsPre[i].x);
				minY = min(minY, pointsPre[i].y);
				curHomography.inliers.push_back(pointsPre[i]);
				curHomography.corresponding.push_back(pointsCur[i]);
				pointsPre.erase(pointsPre.begin() + i);
				pointsCur.erase(pointsCur.begin() + i);
			}
		}
		curHomography.minX = minX;
		curHomography.maxX = maxX;
		curHomography.minY = minY;
		curHomography.maxY = maxY;
		curHomography.H = H;
		curHomography.size = (maxY - minY)*(maxX - minX);
		hvec.push_back(curHomography);
	}
	calDistance(hvec, img.rows*img.cols);

	Mat test;
	test.create(img.rows, img.cols, CV_8UC3);
	img.copyTo(test(Rect(0, 0, img.cols, img.rows)));
	for(int i = 0; i < hvec.size(); i++) {
		int minX = hvec[i].minX, minY = hvec[i].minY, maxX = hvec[i].maxX, maxY = hvec[i].maxY;
		cout << hvec[i].minX << " " << hvec[i].minY << " " << hvec[i].maxX << " " << hvec[i].maxY << endl; 
		line(test, Point(minX, minY), Point(minX, maxY), Scalar(0, 0, 255, 0));
		line(test, Point(minX, minY), Point(maxX, minY), Scalar(0, 0, 255, 0));
		line(test, Point(maxX, minY), Point(maxX, maxY), Scalar(0, 0, 255, 0));
		line(test, Point(minX, maxY), Point(maxX, maxY), Scalar(0, 0, 255, 0));
	}
	imwrite(curDir + "/saliency/Homography_" + to_string(imageID) + ".jpg", test);

	for(int i = 0; i < hvec.size(); i++) {
		cout << hvec[i].aveSaliency << " " << hvec[i].size << endl;
		for(int j = hvec[i].minX; j < hvec[i].maxX; j++) {
			for(int k = hvec[i].minY; k < hvec[i].maxY; k++) {
				tSaliency.at<uchar>(k, j) = max(tSaliency.at<uchar>(k, j), hvec[i].aveSaliency);
			}
		}
	}
	imwrite(curDir + "/saliency/temporal_" + to_string(imageID) + ".jpg", tSaliency);
	imageID++;

	return tSaliency;
} 

Mat temporalSaliency(Mat preImg, Mat img) {
	int minHessian = 400;
	vector<KeyPoint> keyPointsPre, keyPointsCur;
	//SiftDescriptorExtractor detector;
	SurfFeatureDetector detector(minHessian);
	detector.detect(preImg, keyPointsPre);
	detector.detect(img, keyPointsCur);

	SurfDescriptorExtractor extractor;
	//SiftDescriptorExtractor extractor;
	Mat desPre, desCur;
	extractor.compute(preImg, keyPointsPre, desPre);
	extractor.compute(img, keyPointsCur, desCur);
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(desPre, desCur, matches);

	double maxDist = 0, minDist = 100000;
	for(int i = 0; i < matches.size(); i++) {
		double dist = matches[i].distance;
		maxDist = max(maxDist, dist);
		minDist = min(minDist, dist);
	}

	vector<DMatch> goodMatch;
	double gap = maxDist - minDist;
	cout << maxDist << " " << minDist << endl;

	for(int i = 0; i < desPre.rows; i++) {
		if(matches[i].distance < 5 * minDist) {
			goodMatch.push_back(matches[i]);
		}
	}

	vector<Point2f> goodPointsPre;
	vector<Point2f> goodPointsCur;
	for(int i = 0; i < goodMatch.size(); i++) {
		goodPointsPre.push_back(keyPointsPre[goodMatch[i].queryIdx].pt);
		goodPointsCur.push_back(keyPointsCur[goodMatch[i].trainIdx].pt);
	}
	Mat drawMat;
	drawMatch(preImg, img, goodPointsPre, goodPointsCur, drawMat);
	imwrite(curDir + "/saliency/O_" + to_string(imageID) + ".jpg", drawMat);
	Mat ans = getTemporalSaliency(preImg, goodPointsPre, goodPointsCur);
 	return ans;
}

Mat blurSpatialTemporal(Mat sSaliency, Mat tSaliency) {
	Mat ans = Mat(sSaliency.rows, sSaliency.cols, CV_8UC1);
	for(int i = 0; i < sSaliency.rows; i++) {
		for(int j = 0; j < sSaliency.cols; j++) {
			ans.at<uchar>(i, j) = min(sSaliency.at<uchar>(i, j) + 0.1*tSaliency.at<uchar>(i, j), 255);
		}
	}
	return ans;
}

void saliencyDetect(string dir) {
	curDir = dir;
	vector<string> imgPath = getImagePath(dir, 0);
	Mat preImg;
	for(int i = 0; i < imgPath.size(); i++) {
		cout << dir + "/" + imgPath[i] << endl;
		Mat img = imread(dir + "/" + imgPath[i]);
		if(i == 0) {
			Mat sSaliency = spatialSaliency(img.data, img.cols, img.rows, img.cols * img.channels());
			imwrite(dir + "/saliency/spatial_" + to_string(i) + ".jpg", sSaliency);
		} else {
			Mat sSaliency = spatialSaliency(img.data, img.cols, img.rows, img.cols * img.channels());
			Mat tSaliency = temporalSaliency(preImg, img);
			Mat blurSaliency = blurSpatialTemporal(sSaliency, tSaliency);
			imwrite(dir + "/saliency/spatial_" + to_string(i) + ".jpg", sSaliency);
			imwrite(dir + "/saliency/blur_" + to_string(i) + ".jpg", blurSaliency);
		}
		preImg = img;
	}
}