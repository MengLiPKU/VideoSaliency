#include "Saliency.h"

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

void SalientRegionDetectionBasedOnFT(unsigned char *src, unsigned char* &saliencyMap, int width, int height, int stride)
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
	vec2uchar(normaledSalmap, saliencyMap);
}