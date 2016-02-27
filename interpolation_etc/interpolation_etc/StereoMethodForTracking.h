#pragma once
#include "cxcore.h"
//using namespace cv;

class CStereoMethodForTracking
{
public://private:
	cv::Size imgSize;
	cv::Mat * pGrayMat;
	cv::Mat * colorWeight;//待改，因为没几个值
	cv::Mat * FilterMat;

	int interpWinSize;
	double sigma1,sigma2;
public:
	CStereoMethodForTracking();
	CStereoMethodForTracking(cv::Mat & pGrayMat_v, const cv::Size & v_imgSize,int v_interpWinSize,double v_sigma1,double v_sigma2);
	void FastBFilterDepthInterp(double INVAILD);
	
	~CStereoMethodForTracking()
	{
		delete pGrayMat;
		delete colorWeight;
		delete FilterMat;
	}

};

