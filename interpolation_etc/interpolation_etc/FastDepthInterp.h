#pragma once
#include "cxcore.h"
//using namespace cv;

class CFastDepthInterp//CStereoMethodForTracking
{
public://private:
	cv::Size imgSize;
	cv::Mat * pGrayMat;
	cv::Mat * colorWeight;//待改，因为没几个值
	cv::Mat * FilterMat;

	int interpWinSize;
	double sigma1,sigma2;
public:
	CFastDepthInterp();
	CFastDepthInterp(const cv::Mat & pGrayMat_v, const cv::Size & v_imgSize,int v_interpWinSize,double v_sigma1,double v_sigma2);
	void fastBFilterDepthInterp(double INVAILD);


	~CFastDepthInterp()
	{
		if (pGrayMat != NULL) delete pGrayMat;
		if (colorWeight != NULL) delete colorWeight;
		if (FilterMat != NULL) delete FilterMat;
	}

};

