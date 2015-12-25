#include "Saliency.h"
#include "util.h"

int main() {
	string inputPath;
	cin >> inputPath;
	saliencyDetect(inputPath);
	system("pause");
}