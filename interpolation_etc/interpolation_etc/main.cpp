#include <iostream>
#include <stdio.h>
#include "cxcore.h"
#include <highgui.h>  
#include "cvaux.h"
#include "cxmisc.h"
#include "cv.h"   
#include "opencv2/nonfree/features2d.hpp"

#include "StereoMethodForTracking.h"
using namespace cv;
using namespace std;

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
	cv::drawKeypoints(imgL, keypoints, output);
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
	CStereoMethodForTracking smft(disp_coarse,imgL.size(),nWindows,dSigma1,dSigma2);
	smft.FastBFilterDepthInterp(0);

	imshow("0",dispTrue);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	//imshow("3",fineDepthMap/15);
	imshow("4",*(smft.FilterMat)/15);
	waitKey(0);
}