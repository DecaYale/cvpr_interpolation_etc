#pragma once
#include "cxcore.h"
class CSimpleFeatureMatching
{
private:
	//cv::Mat * imgL,*imgR;
	//cv::Mat * gradient_x;

	int m_gradientStep;
	//std::vector<std::vector<int> > & feaList;
	int m_winSize;
	double m_xDiffThresh; //the threshold of the gradient in x direction to decide whether be selected as a condidate or not 
	double m_costThresh;//matching window cost threshold
	double m_peakRatio;//

	void gradient(const cv::Mat & img, cv::Mat & gradient_x,unsigned int s);
	void featureExtract( const cv::Mat & gradient_x,std::vector<std::vector<int> > & feaList, double xDiffThresh);
	double matchCost(const cv::Mat & imgL, const cv::Mat & imgR,int xl,int yl,int xr,int yr,int winSize);
	void featureMatching(const cv:: Mat & imgL,const cv::Mat & imgR,const std::vector<std::vector<int> > & listL,const std::vector<std::vector<int> > & listR,std::vector <std::vector<int> > & pair);
public:
	CSimpleFeatureMatching();
	CSimpleFeatureMatching(int gradientStep,double xDiffThresh, double costThresh,double peakRatio, int winSize)
		:m_gradientStep(gradientStep),m_xDiffThresh(xDiffThresh),m_costThresh(costThresh),m_peakRatio(peakRatio),m_winSize(winSize)
	{

	}
	

	void sparseDisparity(const cv::Mat & imgL,const cv::Mat & imgR, cv::Mat & sparseDisp);


};