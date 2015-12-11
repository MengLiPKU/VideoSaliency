#include "Saliency.h"
#include "util.h"

int main() {
	string inputPath;
	cin >> inputPath;
	vector<string> imagePath = getImagePath(inputPath, 0);
	saliencyDetect(imagePath);
	system("pause");
}