#pragma once

#include "cxcore.h"

extern void BilateralFilter(cv::Mat &pGrayMat, cv::Mat & FilterMat, int nWidth, int nHeight, double dSigma1, double dSigma2, int nWindows); 
extern void BFilterDepthInterp(cv::Mat &pGrayMat, cv::Mat& colorWeight, cv::Mat & FilterMat, int nWidth, int nHeight, double dSigma1, double dSigma2, int nWindows,double INVAILD = 255);
extern void FastBFilterDepthInterp(cv::Mat &pGrayMat, cv::Mat& colorWeight,cv::Mat & FilterMat, int nWidth, int nHeight, double dSigma1, double dSigma2, int nWinWidth,double INVAILD=255);