//#include "StereoMatcher.h"

#include "cxcore.h"
using namespace cv;
double dataCost(double val0,double val1)
{
	return abs(val0 - val1);
}
void RawCost(const cv::Mat & img_L,const cv::Mat & img_R, cv::Mat & dataCostCube,int dLevels)
{
	int width = img_L.cols;
	int height = img_L.rows;
	for(int z=0;z<dLevels;z++)
	{
		for(int y=0;y < height; y++)
		{
			for(int x=0; x<width; x++)
			{
				if (x+z>=width) continue;
				dataCostCube.at<double>(z,y,x) = dataCost( (double)img_R.at<uchar>(y,x),img_L.at<uchar>(y,x+z) );
			}
		}
		
	}
}
//winCostCube:为窗口累加后的cost结果，winWidth 为窗口宽度
//dataCostCube: 窗口累加前的单像素cost结果
void costAggregate(cv::Mat & winCostCube ,cv::Mat & dataCostCube,int height,int width, int dLevels,int winWidth)//void costAggregate(cv::Mat & winCostCube ,const cv::Mat & img_L,const cv::Mat & img_R, cv::Mat & dataCostCube,int dLevels,int winWidth)//winWidth is odd
{
	/*int height = img_L.rows;
	int width = img_L.cols;*/
	//int size_tmp[3];
	//size_tmp[0] = winCostCube.size[0];//winCostCube.step[2]/winCostCube.step[1]/winCostCube.step[0];
	//size_tmp[1] = winCostCube.size[1];
	//size_tmp[1] = winCostCube.size[2];
	cv::Mat tmp(3,winCostCube.size,winCostCube.type(),Scalar(0));

	//暂时只考虑中间部分，边沿部分略去不处理
	for(int d=0;d<dLevels;d++)
	{
		for(int y=0;y<height;y++)
		{
			int sum = 0;
			for(int i = 0;i<winWidth;i++)
			{
				sum += dataCostCube.at<double>(d,y,i);
			}
			for (int x =0;x <width;x++ ) //x 代表窗口中心元素坐标
			{
				if (x-winWidth/2 <= 0 || x+winWidth/2 >= width) continue;
				tmp.at<double>(d,y,x) = sum;
				sum = sum + (- dataCostCube.at<double>(d,y,x-winWidth/2 -1) + dataCostCube.at<double>(d,y,x+winWidth/2) );
				//tmp.at<double>(d,y,x) = sum;
			}
		}
	}

	for(int d=0;d<dLevels;d++)
	{
		for(int x=0;x<width;x++)
		{
			int sum =0;
			for (int i=0;i<winWidth;i++)
			{
				sum += tmp.at<double>(d,i,x);//dataCostCube.at<double>(d,i,x);
			}
			for (int y=0;y<height;y++)
			{
				if (y-winWidth/2 <= 0 || y+winWidth/2 >= height) continue;
				winCostCube.at<double>(d,y,x) = sum;
				sum = sum + (- tmp.at<double>(d,y-winWidth/2 -1,x) + tmp.at<double>(d,y+winWidth/2,x) );
			}
		}
	}

}
void WTA(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat & depthMap,int dLevels ,int winWidth)
{
	int height = img_L.rows;
	int width = img_L.cols;
	int *size = new int[3];
	size[0] = dLevels; size[1] = height; size[2] = width;
	 double val = 1E6;	//大了会出问题，why？
	cv::Mat * dataCostCube = new Mat(3,size,CV_64FC1,Scalar(val) );
	cv::Mat * winCostCube = new Mat(3,size,CV_64FC1,Scalar(0) );

	RawCost(img_L,img_R, *dataCostCube,dLevels);
	costAggregate(*winCostCube, *dataCostCube, height, width, dLevels, winWidth);

	for (int y=0; y<height; y++)
	{
		for (int x=0; x<width; x++)
		{
			double costMin = 1E20;
			double di = 0;
			for (int d = 0;d<dLevels;d++)
			{
				double cost = (*winCostCube).at<double>(d,y,x);

				if ( cost < costMin )
				{
					costMin = cost;
					di = d;
				
				}
			}
			/*if (di!=0)
			{
				static int n=0;
				n++;
				printf("%d\n",n);
			}*/
			depthMap.at<double>(y,x) = di;

		}
	}
	delete [] size; size = NULL;
	delete dataCostCube; dataCostCube =NULL;
	delete winCostCube;  winCostCube =NULL;
}
