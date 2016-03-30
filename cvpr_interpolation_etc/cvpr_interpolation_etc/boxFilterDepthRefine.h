#pragma once

#include "cxcore.h"
extern void boxFilterStereo(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat &fineDepthMap, int dLevels ,int winWidth);
extern void integralImgCal(double * originImg, int height,int width,int widthStep);
extern void boxFilterDepthRefine_prototype(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat &coarseDepthMap, cv::Mat &fineDepthMap, int dLevels ,int winWidth,int deviation);
extern void boxFilterDepthRefine(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat &coarseDepthMap,const std::vector<bool> & dispSample, cv::Mat &fineDepthMap,cv::Mat & isValidate, int dLevels ,int winWidth,int deviation,double validThreshold = 5,double curvThreshold = -2,double peakRatio = 1.1);
extern cv::Mat *  boxFilterDepthRefine_test(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat &coarseDepthMap, const std::vector<bool> & dispSample, cv::Mat &fineDepthMap,cv:: Mat & isValid, int dLevels ,int winWidth,int deviation,double validThreshold = 5,double curvThreshold = -2,double peakRatio = 1.1);