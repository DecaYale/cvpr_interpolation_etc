#pragma once
#include<string>

class CxImageFuns{
public:
	static int GetImageFileType(const std::string& name);
private:
	static std::string FindExtension(const std::string& name);
	static int FindFormat(const std::string& ext);
};

