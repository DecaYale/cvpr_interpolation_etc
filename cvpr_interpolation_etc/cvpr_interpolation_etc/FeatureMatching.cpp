
#include "FeatureMatching.h"
#include "cxcore.h"
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
			 
			 gradient_x.at<double>(i,j) = ( img.at<uchar>(i,j+s) - img.at<uchar>(i,j-s) )/(2*s);

		}
	}

}

void featureExtract( const Mat & gradient_x,vector<vector<int> > & feaList, double xDiffThresh)
{
	int H = gradient_x.rows;
	int W = gradient_x.cols;
	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			if (gradient_x.at<double>(i,j)>xDiffThresh )
			{
				feaList[i].push_back(j);
			}
		}
	}
}

double matchCost(const Mat & imgL, const Mat & imgR,int xl,int yl,int xr,int yr,int winSize)
{
	
	double cost = 0;
	int halfWinSize = (int)winSize/2;
	if (yl<halfWinSize || yl>=imgL.rows-halfWinSize || xl <halfWinSize || xl>=imgL.cols-halfWinSize
		||yr<halfWinSize|| yr>imgR.rows-halfWinSize || xr <halfWinSize || xr>=imgR.cols-halfWinSize)
		continue;


	for(int i=0; i<winSize; i++)
	{
		for(int j=0; j<winSize; j++)
		{
			int yL = yl-halfWinSize + i;
			int xL = xl-halfWinSize + j;
			int yR = yr-halfWinSize + i;
			int xR = xr-halfWinSize + j;

			cost += abs(imgL.at<uchar>(yL,xL) - imgR.at<uchar>(yR,xR) );

		}
	}
	cost /= winSize * winSize;
	return cost;
}
void featureMatching(const Mat & imgL,const Mat & imgR,vector<vector<int> > listL,vector<vector<int> > & listR,int winSize)
{
	int H = imgL.rows;
	int W = imgL.cols;
 
	assert (listL.size() == H && listR.size() ==H);
	vector<vector<double> > dist(1);
	for(int y=0; y<H; y++)
	{
		int Nl = listL[y].size();
		int Nr = listR[y].size();
		
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

				dist[l][r] = matchCost(imgL, imgR,xl,y,xr,y,winSize);
			}
		}

		//matching 


	}
	

}
void sparseDisparity()
{
	Mat imgLGradient(imgL.size(),CV_64FC1,Scalar(0));
	Mat imgRGradient(imgR.size(),CV_64FC1,Scalar(0));

	//计算x方向梯度
	unsigned int s = 2;
	gradient(imgL, imgLGradient,s);
	gradient(imgL, imgRGradient,s);

	//

	featureExtract( imgLGradient,vector<vector<int> > & listL, double xDiffThresh);
	featureExtract( imgRGradient,vector<vector<int> > & listR, double xDiffThresh);
}
