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

