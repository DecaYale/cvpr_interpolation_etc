
#include "FeatureMatching.h"
#include "cxcore.h"
#include <highgui.h>  
using namespace cv;

//////////////////////////////////////////////////////////////////////////
//img: uchar
//gradient_x：double
void gradient(const Mat & img, Mat & gradient_x,unsigned int s)
{
	if (img.size() != gradient_x.size()) gradient_x.create(img.size(),CV_64FC1);

	int imgH = img.rows;
	int imgW = img.cols;

	for(int i=0; i<imgH; i++)
	{
		for(int j=0; j<imgW; j++)
		{
			 if (j<s || j>=imgW - s) continue;
			// double test =  ( (double) img.at<uchar>(i,j+s) - (double) img.at<uchar>(i,j-s) )/(2*s);
			 gradient_x.at<double>(i,j) = ( (double) img.at<uchar>(i,j+s) - (double) img.at<uchar>(i,j-s) )/(2*s);
			 
		
		}
	}

}

void featureExtract( const Mat & gradient_x,vector<vector<int> > & feaList, double xDiffThresh)
{
	int H = gradient_x.rows;
	int W = gradient_x.cols;

	feaList.clear();
	feaList.resize(H);

//	int forTest = 0;///////
	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			if (abs( gradient_x.at<double>(i,j) ) > xDiffThresh )
			{
				feaList[i].push_back(j);
				//forTest++;/////////
			}
		}
	}
	//printf("%d\n",forTest);
}

double matchCost(const Mat & imgL, const Mat & imgR,int xl,int yl,int xr,int yr,int winSize)
{
	
	double cost = 0;
	int halfWinSize = (int)winSize/2;
	if (yl<halfWinSize || yl>=imgL.rows-halfWinSize || xl <halfWinSize || xl>=imgL.cols-halfWinSize
		||yr<halfWinSize|| yr>imgR.rows-halfWinSize || xr <halfWinSize || xr>=imgR.cols-halfWinSize)
		return 1e10;


	for(int i=0; i<winSize; i++)
	{
		for(int j=0; j<winSize; j++)
		{
			int yL = yl-halfWinSize + i;
			int xL = xl-halfWinSize + j;
			int yR = yr-halfWinSize + i;
			int xR = xr-halfWinSize + j;

			cost += abs((double)imgL.at<uchar>(yL,xL) - imgR.at<uchar>(yR,xR) );

		}
	}
	cost /= (winSize * winSize);
	return cost;
}
void featureMatching(const Mat & imgL,const Mat & imgR,const vector<vector<int> > & listL,const vector<vector<int> > & listR,vector <vector<int> > & pair,double costThresh,double peakRatio, int winSize)
{
	int H = imgL.rows;
	int W = imgL.cols;
	int halfWinSize = (int)winSize/2;
	pair.clear();//清理

	assert (listL.size() == H && listR.size() == H );
	vector<vector<double> > dist(1);
	 //存储匹配点对[xl,yl,xr,yr;xl,yl,xr,yr;xl,yl,xr,yr...]
	for(int y=0; y<H; y++)
	{
		int Nl = listL[y].size();
		int Nr = listR[y].size();
		
		if (Nl == 0 || Nr ==0) continue;
	
		// 检查dist大小
		if (dist.size()<Nl)
		{
			dist.resize(2*Nl);
			int w = dist[0].size();
			for(int i=0;i<dist.size();i++)
			{
				dist[i].resize(w);
			}
		}
		if (dist[0].size()<Nr)
		{
			for(int i=0;i<dist.size();i++)
			{
				dist[i].resize(2*Nr);
			}
		}
		

		for(int l=0; l<Nl;l++)
		{
			for(int r=0; r<Nr; r++)
			{
				int xl = listL[y][l];
				int xr = listR[y][r];

				/*if (y<halfWinSize || y>=H-halfWinSize ) 
				{
					Nr = 0;
					Nl = 0;
					break;
				}
				else if (xl <halfWinSize || xl>=W-halfWinSize)
				{
					Nl --;
					continue;
				}
				else if ( xr <halfWinSize || xr>=W-halfWinSize)
				{

					Nr--;
					continue;
				}*/


				dist[l][r] = matchCost(imgL, imgR,xl,y,xr,y,winSize);
			}
		}
		/*if (y==98)
		{
			for(int l=0;l<Nl;l++)
			{
				for(int r=0;r<Nr;r++)
					printf("%.2f" ,dist[l][r]);
				printf("\n");
			}
		}*/
		//每行特征点 matching 
		vector<vector<int> > lrMinList(Nl); //Nl x 2 ，第i行第一个元素对应 距离图像第y行 第i个元素最近的右图像的像素索引
		vector<vector<int> > rlMinList(Nr);

		for(int l=0; l<Nl; l++)
		{
			double minV = 1e10;
			double min2V = 1e10;
			int minIdx = -1;
			int min2Idx = -1;
			for(int r=0; r<Nr; r++)
			{
				//找到对应每个l的最小的两个对应的r
				double dist_lr = dist[l][r];
				if ( dist_lr < minV ) 
				{
					//更新次小的
					min2Idx = minIdx;
					min2V = minV;
					//更新最小的
					minIdx = r;
					minV = dist_lr;
					
					
				}
				else if (dist_lr < min2V) 
				{
					min2Idx = r;
					min2V = dist_lr;
				}
				
			}
			//if (minIdx != -1)
				lrMinList[l].push_back(minIdx);//将与左边第y行，第l个特征点最接近的 右边第y行的特征点保存
			//if (min2Idx != -1)
				lrMinList[l].push_back(min2Idx);//将与左边第y行，第l个特征点次接近的 右边第y行的特征点保存

		}

		for(int r=0; r<Nr; r++)
		{
			double minV = 1e10;
			double min2V = 1e10;
			int minIdx= -1;
			int min2Idx = -1;
			for(int l=0; l<Nl; l++)
			{
				//找到对应每个l的最小的两个对应的r
				double dist_rl = dist[l][r];//double dist_rl = dist[r][l];
				if ( dist_rl < minV ) 
				{
					//更新次小的
					min2Idx = minIdx;
					min2V = minV;
					//更新最小的
					minIdx = l;
					minV = dist_rl;
				}
				else if (dist_rl < min2V) 
				{
					min2Idx = l;
					min2V = dist_rl;
				}

			}
			//if (minIdx != -1)
				rlMinList[r].push_back(minIdx);
			//if (min2Idx != -1)
				rlMinList[r].push_back(min2Idx);

		}
		//交叉验证 + peak ratio?

		for(int l=0; l<Nl; l++)
		{
			vector<int> tmp;//[xl,yl,xr,yr]
			int lLover = lrMinList[l][0];

			if (lLover == -1) continue;

			int rLover = rlMinList[lLover][0];

			if (rLover == -1) continue;
			//if (lLover == -1 || rLover ==-1) continue;


			if(l == rLover && dist[l][lLover]< costThresh) //	&& dist[l][lLover] < 11
			{
				if (lrMinList[l][1] != -1 && dist[l][lrMinList[l][1]]/dist[l][lLover]  < peakRatio ) continue;

				tmp.push_back(listL[y][l] );
				tmp.push_back(y);
				tmp.push_back(listR[y][lLover]);
				tmp.push_back(y);
				//存入pair
				pair.push_back(tmp);
			}
		}
		

	}
	

}
void sparseDisparity(const Mat & imgL,const Mat & imgR, Mat & sparseDisp, double xDiffThresh, double costThresh,double peakRatio, int winSize)
{
	Mat imgLGradient(imgL.size(),CV_64FC1,Scalar(0));
	Mat imgRGradient(imgR.size(),CV_64FC1,Scalar(0));

	//计算x方向梯度
	unsigned int s = 2;
	gradient(imgL, imgLGradient,s);
	gradient(imgR, imgRGradient,s);

	//
	vector< vector<int>> listL,listR;

	featureExtract( imgLGradient,listL, xDiffThresh);
	featureExtract( imgRGradient,listR, xDiffThresh);

	
	vector <vector<int> > pair;

	featureMatching(imgL,imgR,listL, listR, pair, costThresh, peakRatio, winSize);

	if (sparseDisp.size != imgL.size) sparseDisp.create(imgL.size(),CV_32SC1);
	int H = sparseDisp.rows;
	int W = sparseDisp.cols;
	for(int i=0; i<pair.size(); i++)
	{
		int y = pair[i][1];
		int x = pair[i][0];
		int disp = pair[i][0] - pair[i][2];

		sparseDisp.at<double>(y,x) =  disp>0 ? disp:0;
	}


	//for test
	FILE * fp = fopen("pair.txt","w+");
	for(int i=0;i<pair.size();i++)
	{
		fprintf(fp,"%d %d %d %d\n",pair[i][0],pair[i][1],pair[i][2],pair[i][3]);

	}
	fclose(fp);

	Mat imgC(imgL.rows,2*imgL.cols,imgL.type());
	//imgC( Range(1,imgL.rows), Range(1,imgL.cols) ) = imgL;//.clone();
	//imgC( Range(1,imgL.rows), Range(imgL.cols,2*imgL.cols) ) = imgR;//.clone();
	for(int i=0;i<imgC.rows;i++)
	{
		for(int j=0;j<imgL.cols;j++)
		{
			imgC.at<uchar>(i,j) = imgL.at<uchar>(i,j);
			imgC.at<uchar>(i,j+imgL.cols) = imgR.at<uchar>(i,j);
		}
		
	}
	int size = pair.size();
	for(int i=0; i<pair.size();i++)
	{
		//srand(time(0))
		circle(imgC,Point(pair[i][0],pair[i][1]),2,Scalar(0,0,0));
		circle(imgC,Point(pair[i][2]+imgL.cols,pair[i][3]),2,Scalar(100,200,200));
		line(imgC,Point(pair[i][0],pair[i][1]),Point(pair[i][2]+imgL.cols,pair[i][3]),Scalar(255,255,255));

	}
	imshow("1",imgC);
	waitKey(0);

}
