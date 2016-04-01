#include <iostream>
#include <stdio.h>
#include "cxcore.h"
#include <highgui.h>  
#include "cvaux.h"
#include "cxmisc.h"
#include "cv.h"   
#include "opencv2/nonfree/features2d.hpp"

#include "FastDepthInterp.h"
#include "BoxFilterDepthRefine.h"
#include "SimpleFeatureMatching.h"
#include "Nonlocal/Nonlocal.h"
using namespace cv;
using namespace std;


clock_t timer;




#if 1
int main()
{
	const cv::Mat imgL = cv::imread ("./data/scene1.row3.col3.ppm", 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("./data/scene1.row3.col4.ppm", 0);//("./data/scene1.row3.col4.ppm", 0);

	//cv::Mat dispTrue = cv::imread ("./data/truedisp.row3.col3.pgm", 0);//("./data/truedisp.row3.col3.pgm", 0);//("./data2/disp1_half.png", 0);//("./data/truedisp.row3.col3.pgm", 0);
	//dispTrue = dispTrue/15;///15;//�����dispTrue�����ǵ��������

	////sift ��������㲢����Ӧ���ͼ�ϲ������disp_coarse
	//cv::SiftFeatureDetector detector;
	//std::vector<cv::KeyPoint> keypoints;
	//detector.detect(imgL, keypoints);

	// Add results to image and save.
	//cv::Mat output;
	//cv::drawKeypoints(imgL, keypoints, output);
	//cv::imwrite("sift_result.jpg", output);
	
	int gradientStep  = 2;
	double xDiffThresh = 12 ;
	double costThresh = 10;
	double peakRatio = 3;//cm2/cm1
	int winSize = 5;
	Mat sparseDisp(imgL.size(),CV_64FC1,Scalar(0));
timer =clock();
	CSimpleFeatureMatching sfm( gradientStep, xDiffThresh, costThresh,peakRatio, winSize);
	sfm.sparseDisparity(imgL,imgR,sparseDisp);
cout<<clock()-timer<<endl;



	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//��ʱ����CV_8UC1

	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 51;//˫���˲����ڴ�С
	
	//���ٲ�ֵ
timer =clock();
	CFastDepthInterp smft(sparseDisp,imgL.size(),nWindows,dSigma1,dSigma2);
	smft.fastBFilterDepthInterp(0);
cout<<clock()-timer<<endl;
	//boxFiler refine 
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	int dLevels = 20;
timer = clock();
	CBoxFilterDepthRefine bfdr(imgL,imgR,dLevels);
	bfdr.boxFilterDepthRefine(sparseDisp,fineDepthMap);
cout<<clock()-timer<<endl;
	

Nonlocal nl;
double sigma = 0.5;

Mat costCube;
bfdr.getDataCostCube(costCube);
Mat depthResult(imgL.size(),CV_64FC1,Scalar(0));
nl.stereo(imgL, costCube, depthResult, sigma,0);//nl.stereo(imgL,* winCostCube,depthResult, sigma,0);



//for test
Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
Mat diffImg(imgL.size(),CV_64FC1,Scalar(0));
Mat confidenceMap;
bfdr.getConfidenceMap(confidenceMap);

for(int i =0;i<fineDepthMap.rows;i++)
	for(int j=0;j<fineDepthMap.cols;j++)
	{
		subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);
		diffImg.at<double>(i,j) = ( confidenceMap.at<uchar>(i,j)==1 ? abs( 2* fineDepthMap.at<double>(i,j) - confidenceMap.at<uchar>(i,j) ):0 ); 
	}

	//imshow("0",dispTrue);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	imshow("3",fineDepthMap/15);
	imshow("4",*(smft.FilterMat)/15);
	imshow("5",confidenceMap*255);
	imshow("6",subtractImg/20);
	imshow("7",depthResult/20);
	waitKey(0);
}
#elif 1 
int main()
{
	const cv::Mat imgL = cv::imread ("./data/scene1.row3.col3.ppm", 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("./data/scene1.row3.col4.ppm", 0);//("./data/scene1.row3.col4.ppm", 0);

	cv::Mat dispTrue = cv::imread ("./data/truedisp.row3.col3.pgm", 0);//("./data/truedisp.row3.col3.pgm", 0);//("./data2/disp1_half.png", 0);//("./data/truedisp.row3.col3.pgm", 0);
	dispTrue = dispTrue/15;///15;//�����dispTrue�����ǵ��������

	//sift ��������㲢����Ӧ���ͼ�ϲ������disp_coarse
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(imgL, keypoints);

	// Add results to image and save.
	cv::Mat output;
	//cv::drawKeypoints(imgL, keypoints, output);
	//cv::imwrite("sift_result.jpg", output);

	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//��ʱ����CV_8UC1

	//��ϡ����Ȳ�ֵ�õ�������� 
	for (int i=0;i<keypoints.size();i++)
	{
		disp_coarse.at<double>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x) = dispTrue.at<uchar>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x);
	}
	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 51;//˫���˲����ڴ�С
	//cv::Mat FilterMat(disp_coarse.size(),CV_64FC1,Scalar(0));
	//cv::Mat colorWeight(disp_coarse.size(),CV_64FC1,Scalar(0));//ע����Ч�㴦Ϊ0����������

	//���ٲ�ֵ
timer =clock();
	CFastDepthInterp smft(disp_coarse,imgL.size(),nWindows,dSigma1,dSigma2);
	smft.fastBFilterDepthInterp(0);
cout<<clock()-timer<<endl;
	//boxFiler refine 
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	int dLevels = 20;
timer = clock();
	CBoxFilterDepthRefine bfdr(imgL,imgR,dLevels);
	bfdr.boxFilterDepthRefine(disp_coarse,fineDepthMap);
cout<<clock()-timer<<endl;
	

Nonlocal nl;
double sigma = 0.1;
//int fortest = winCostCube.size[0];
Mat costCube;
bfdr.getDataCostCube(costCube);
Mat depthResult(imgL.size(),CV_64FC1,Scalar(0));
nl.stereo(imgL, costCube, depthResult, sigma,0);//nl.stereo(imgL,* winCostCube,depthResult, sigma,0);



//for test
Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
Mat diffImg(imgL.size(),CV_64FC1,Scalar(0));
Mat confidenceMap;
bfdr.getConfidenceMap(confidenceMap);

for(int i =0;i<fineDepthMap.rows;i++)
	for(int j=0;j<fineDepthMap.cols;j++)
	{
		subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);
		diffImg.at<double>(i,j) = ( confidenceMap.at<uchar>(i,j)==1 ? abs( 2* fineDepthMap.at<double>(i,j) - confidenceMap.at<uchar>(i,j) ):0 ); 
	}

	imshow("0",dispTrue);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	imshow("3",fineDepthMap/15);
	imshow("4",*(smft.FilterMat)/15);
	imshow("5",confidenceMap*255);
	imshow("6",subtractImg/20);
	imshow("7",depthResult/20);
	waitKey(0);
}

#endif 