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

#include "LocalStereoMatching.h"
using namespace cv;
using namespace std;


clock_t timer;




#if 1
int main()
{
	//
	char root[] = "E:/ExpData/stereo/data";
	char dirL[100];
	char dirR[100];
	char dirDispF[100];
	char dirDispS[100];
	int i=2;
	sprintf(dirL,"%s%d%s",root,i,"/im0.png");
	sprintf(dirR,"%s%d%s",root,i,"/im1.png");

	
	sprintf(dirDispF,"%s%d%s",root,i,"/DispF.png");
	sprintf(dirDispS,"%s%d%s",root,i,"/DispS.png");


 	const cv::Mat imgL = cv::imread (dirL, 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread (dirR, 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	cout<<dirL<<endl;

	/*imshow("test",imgL);
	waitKey(0);*/
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



	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//暂时先是CV_8UC1

	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 71;//双边滤波窗口大小
	
	//快速插值
timer =clock();
	CFastDepthInterp fdi(sparseDisp,imgL.size(),nWindows,dSigma1,dSigma2);
	fdi.fastBFilterDepthInterp(0);
	//fdi.getFilterImg()
cout<<clock()-timer<<endl;

	//boxFiler refine 
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	int dLevels = 100;

timer = clock();
//CLocalStereoMatching(const cv::Mat & imgL,const cv::Mat & imgR,
//	int dLevels, int winWidth = 5,int deviation = 10, 
//	double validThreshold = 10, double curvThreshold = -0.1, double peakRatio = 1.1)
int winWidth = 20;
int deviation = 50;
double validThreshold = 10;
//double curvThreshold = -0.8, double peakRatio = 1.1
CLocalStereoMatching lsm(imgL,imgR,dLevels); //(const cv::Mat & imgL,const cv::Mat & imgR,

	lsm.setDeviation(50);
	lsm.boxFilterDepthRefine(*(fdi.FilterMat),fineDepthMap);
cout<<"?"<<clock()-timer<<endl;
	

//	Nonlocal nl;
//	double sigma = 0.05;
//
//timer = clock();
//	Mat costCube;
//	lsm.getDataCostCube(costCube);
//	Mat depthResult(imgL.size(),CV_64FC1,Scalar(0));
//	nl.stereo(imgL, costCube, depthResult, sigma,0);//nl.stereo(imgL,* winCostCube,depthResult, sigma,0);
//cout<<clock()-timer<<endl;


//for test
Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
Mat diffImg(imgL.size(),CV_64FC1,Scalar(0));
Mat confidenceMap;
lsm.getConfidenceMap(confidenceMap);

for(int i =0;i<fineDepthMap.rows;i++)
	for(int j=0;j<fineDepthMap.cols;j++)
	{
		//subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);
		subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j) : 0);//depthResult.at<double>(i,j)
		diffImg.at<double>(i,j) = ( confidenceMap.at<uchar>(i,j)==1 ? abs( 2* fineDepthMap.at<double>(i,j) - confidenceMap.at<uchar>(i,j) ):0 ); 
	}

	imwrite(dirDispF,fineDepthMap*2);//imwrite("data3/dispF.jpg",fineDepthMap);
	imwrite(dirDispS,subtractImg*2);//imwrite("data3/dispS.jpg",subtractImg);


	imshow("0",sparseDisp/60);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	imshow("3",fineDepthMap/100);
	imshow("4",*(fdi.FilterMat)/100);
	imshow("5",confidenceMap*255);
	imshow("6",subtractImg/100);
	//imshow("7",depthResult/100);
	//imshow("8",diffImg);
	waitKey(0);
}
#elif 1
int main()
{
	//nonlocal + WTA test
	char root[] = "e:/MyDocument/klive sync/计算机视觉/Stereo_Matching/Data Set/Middlebury2006/half size/data";
	char dirL[100];
	char dirR[100];
	char dirDispF[100];
	char dirDispS[100];
	sprintf(dirL,"%s%d%s",root,2,"/view1.png");
	sprintf(dirR,"%s%d%s",root,2,"/view5.png");

	int i=0;
	sprintf(dirDispF,"%s%d%s",root,i,"/DispF.png");
	sprintf(dirDispS,"%s%d%s",root,i,"/DispS.png");


 	const cv::Mat imgL = cv::imread (dirL, 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread (dirR, 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	cout<<dirL<<endl;

	//imshow("test",imgL);
	//waitKey(0);
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



	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//暂时先是CV_8UC1

	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 71;//双边滤波窗口大小
	
	//快速插值
timer =clock();
	CFastDepthInterp fdi(sparseDisp,imgL.size(),nWindows,dSigma1,dSigma2);
	fdi.fastBFilterDepthInterp(0);
	//fdi.getFilterImg()
cout<<clock()-timer<<endl;

	//boxFiler refine 
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	int dLevels = 100;
timer = clock();
CLocalStereoMatching lsm(imgL,imgR,dLevels); //(const cv::Mat & imgL,const cv::Mat & imgR,

	lsm.setDeviation(50);
	lsm.boxFilterDepthRefine(*(fdi.FilterMat),fineDepthMap);
cout<<clock()-timer<<endl;
	

	Nonlocal nl;
	double sigma = 0.05;

timer = clock();
	Mat costCube;
	lsm.getDataCostCube(costCube);
	Mat depthResult(imgL.size(),CV_64FC1,Scalar(0));
	nl.stereo(imgL, costCube, depthResult, sigma,0);//nl.stereo(imgL,* winCostCube,depthResult, sigma,0);
cout<<clock()-timer<<endl;


//for test
Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
Mat diffImg(imgL.size(),CV_64FC1,Scalar(0));
Mat confidenceMap;
lsm.getConfidenceMap(confidenceMap);

for(int i =0;i<fineDepthMap.rows;i++)
	for(int j=0;j<fineDepthMap.cols;j++)
	{
		//subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);
		subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j) : 0);//depthResult.at<double>(i,j)
		diffImg.at<double>(i,j) = ( confidenceMap.at<uchar>(i,j)==1 ? abs( 2* fineDepthMap.at<double>(i,j) - confidenceMap.at<uchar>(i,j) ):0 ); 
	}

	imwrite(dirDispF,fineDepthMap*2);//imwrite("data3/dispF.jpg",fineDepthMap);
	imwrite(dirDispS,subtractImg*2);//imwrite("data3/dispS.jpg",subtractImg);


	imshow("0",sparseDisp/60);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	imshow("3",fineDepthMap/100);
	imshow("4",*(fdi.FilterMat)/100);
	imshow("5",confidenceMap*255);
	imshow("6",subtractImg/100);
	imshow("7",depthResult/100);
	//imshow("8",diffImg);
	waitKey(0);
}

#elif 1
int main()
{
	//测试 Nonlocal + 原有的稀疏插稠密WTA
	char root[] = "e:/MyDocument/klive sync/计算机视觉/Stereo_Matching/Data Set/Middlebury2006/half size/data";
	char dirL[100];
	char dirR[100];
	sprintf(dirL,"%s%d%s",root,2,"/view1.png");
	sprintf(dirR,"%s%d%s",root,2,"/view5.png");


 	const cv::Mat imgL = cv::imread (dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	cout<<dirL;

	//imshow("test",imgL);
	//waitKey(0);
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



	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//暂时先是CV_8UC1

	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 71;//双边滤波窗口大小
	
	//快速插值
timer =clock();
	CFastDepthInterp fdi(sparseDisp,imgL.size(),nWindows,dSigma1,dSigma2);
	fdi.fastBFilterDepthInterp(0);
	//fdi.getFilterImg()
cout<<clock()-timer<<endl;
	//boxFiler refine 
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	int dLevels = 100;
timer = clock();
CBoxFilterDepthRefine bfdr(imgL,imgR,dLevels,20); //(const cv::Mat & imgL,const cv::Mat & imgR,

	bfdr.setDeviation(100);
	bfdr.boxFilterDepthRefine(*(fdi.FilterMat),fineDepthMap);
cout<<clock()-timer<<endl;
	

	Nonlocal nl;
	double sigma = 0.05;

timer = clock();
	Mat costCube;
	bfdr.getDataCostCube(costCube);
	Mat depthResult(imgL.size(),CV_64FC1,Scalar(0));
	nl.stereo(imgL, costCube, depthResult, sigma,0);//nl.stereo(imgL,* winCostCube,depthResult, sigma,0);
cout<<clock()-timer<<endl;


//for test
Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
Mat diffImg(imgL.size(),CV_64FC1,Scalar(0));
Mat confidenceMap;
bfdr.getConfidenceMap(confidenceMap);

for(int i =0;i<fineDepthMap.rows;i++)
	for(int j=0;j<fineDepthMap.cols;j++)
	{
		//subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);
		subtractImg.at<double>(i,j) = (confidenceMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j) : depthResult.at<double>(i,j));//depthResult.at<double>(i,j)
		diffImg.at<double>(i,j) = ( confidenceMap.at<uchar>(i,j)==1 ? abs( 2* fineDepthMap.at<double>(i,j) - confidenceMap.at<uchar>(i,j) ):0 ); 
	}

	imshow("0",sparseDisp/60);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	imshow("3",fineDepthMap/100);
	imshow("4",*(fdi.FilterMat)/100);
	imshow("5",confidenceMap*255);
	imshow("6",subtractImg/100);
	imshow("7",depthResult/100);
	//imshow("8",diffImg);
	waitKey(0);
}
#elif 1 
int main()
{
	const cv::Mat imgL = cv::imread ("./data/scene1.row3.col3.ppm", 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("./data/scene1.row3.col4.ppm", 0);//("./data/scene1.row3.col4.ppm", 0);

	cv::Mat dispTrue = cv::imread ("./data/truedisp.row3.col3.pgm", 0);//("./data/truedisp.row3.col3.pgm", 0);//("./data2/disp1_half.png", 0);//("./data/truedisp.row3.col3.pgm", 0);
	dispTrue = dispTrue/15;///15;//处理后dispTrue是真是的深度数据

	//sift 获得特征点并在相应深度图上采样获得disp_coarse
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(imgL, keypoints);

	// Add results to image and save.
	cv::Mat output;
	//cv::drawKeypoints(imgL, keypoints, output);
	//cv::imwrite("sift_result.jpg", output);

	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//暂时先是CV_8UC1

	//由稀疏深度插值得到稠密深度 
	for (int i=0;i<keypoints.size();i++)
	{
		disp_coarse.at<double>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x) = dispTrue.at<uchar>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x);
	}
	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 51;//双边滤波窗口大小
	//cv::Mat FilterMat(disp_coarse.size(),CV_64FC1,Scalar(0));
	//cv::Mat colorWeight(disp_coarse.size(),CV_64FC1,Scalar(0));//注意无效点处为0，否则会出错。

	//快速插值
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