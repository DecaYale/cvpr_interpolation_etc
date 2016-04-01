#pragma once 
#include "cxcore.h"
class CBoxFilterDepthRefine
{
private:

	cv::Mat * imgL, *imgR;

	int m_dLevels;
	int m_winWidth;
	int m_deviation;
	double m_validThreshold;
	double m_curvThreshold;
	double m_peakRatio;

	cv::Mat * m_dataCostCube;
	cv::Mat * m_winCostCube ;
	cv::Mat * m_confidenceMap;
	cv::Mat * m_dataCostCubeIntegral;

	void offsetGenerate(const cv::Mat& dispMap,cv::Mat & offsetImg);
	void integralImgCal(double * originImg, int height,int width,int widthStep);
public:
	CBoxFilterDepthRefine();
	CBoxFilterDepthRefine(const cv::Mat & imgL,const cv::Mat & imgR,
						int dLevels, int winWidth = 5,int deviation = 10, 
						double validThreshold = 10, double curvThreshold = -2, double peakRatio = 1.2):
						m_dLevels(dLevels),m_winWidth(winWidth),m_deviation(deviation),
						m_validThreshold(validThreshold),m_curvThreshold(curvThreshold),m_peakRatio(peakRatio)
	{
		this->imgL = new cv::Mat();
		this->imgR = new cv::Mat();
		* (this->imgL) = imgL.clone();
		* (this->imgR) = imgR.clone();

		m_confidenceMap = new cv::Mat(this->imgL->size(),CV_8UC1,cv::Scalar(0));
	}

	~CBoxFilterDepthRefine()
	{
		if (imgL != NULL) delete imgL;
		if (imgR != NULL) delete imgR;
		if (m_dataCostCube != NULL) delete m_dataCostCube;
		if (m_winCostCube != NULL) delete m_winCostCube;
		if (m_confidenceMap != NULL) delete m_confidenceMap;
	}
	

	void boxFilterDepthRefine(const cv::Mat &coarseDepthMap,cv:: Mat & fineDepthMap);
	void getDataCostCube(cv::Mat & data_cost_cube);
	void getConfidenceMap(cv::Mat & confidence_map);

};