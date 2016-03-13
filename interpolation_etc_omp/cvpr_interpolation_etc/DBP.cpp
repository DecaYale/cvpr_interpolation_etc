#include "ExWRegularGridBP.h"
#include "DBP.h"
#include <iostream>
#define  INF  1e20
using namespace std;

namespace NumericalAlgorithm{
//////////////////////////////////////////////////////////////////////////
//参数说明
//input:dataCost:左右图像disp 的 raw cost
//		offsetImg: 
//		wImg:权重map
//		truncImg:截止map
//output:labelImg:输出最优的label，disp = labelimg + offsetImg
//////////////////////////////////////////////////////////////////////////
void CDBP::DSolve(ZCubeFloatImage& dataCost, ZIntImage& labelImg, ZIntImage& offsetImg, ZIntImage& isValidImg,ZFloatImage& wImg, ZFloatImage& truncImg)
	{
		int iMaxLabel = dataCost.GetChannel();
		VALUES = iMaxLabel;

		ZCubeFloatImage uI,dI,rI,lI;
		int iWidth = dataCost.GetWidth();
		int iHeight = dataCost.GetHeight();
		

		uI.CreateAndInit(iWidth,iHeight,iMaxLabel);
		dI.CreateAndInit(iWidth,iHeight,iMaxLabel);
		rI.CreateAndInit(iWidth,iHeight,iMaxLabel);
		lI.CreateAndInit(iWidth,iHeight,iMaxLabel);


		DBp_CP(uI, dI, rI, lI, dataCost, offsetImg, wImg, truncImg);

		labelImg.Create(iWidth,iHeight);
		isValidImg.CreateAndInit(iWidth,iHeight,1,0);//added by dy 

		for (int y = 1; y < iHeight-1; y++) {
			for (int x = 1; x < iWidth-1; x++) {
				// keep track of best value for current pixel
				int best = 0;
				float best_val = INF;
				double average = 0;//added by dy
				for (int value = 0; value < iMaxLabel; value++) {
					float val = 
						uI.at( x, y+1,value) +
						dI.at( x, y-1,value) +
						lI.at( x+1, y,value) +
						rI.at( x-1, y,value) +					
						dataCost.at(x,y,value);
					if (val < best_val) {
						best_val = val;
						best = value;
						labelImg.at(x,y) = best;					
					}
				
					average += val;	//added by dy
				}
				average /= iMaxLabel;//added by dy
				////added by dy 第二次遍历求方差或标准差
				//double var = 0;
				//for (int value = 0; value < iMaxLabel; value++) {
				//	float val = 
				//		uI.at( x, y+1,value) +
				//		dI.at( x, y-1,value) +
				//		lI.at( x+1, y,value) +
				//		rI.at( x-1, y,value) +					
				//		dataCost.at(x,y,value);				
				//	var += abs(val-average);
				//}
				//var /= iMaxLabel;
				if (best_val <IS_VALID_THRESHOLD)//if (best_val<IS_VALID_THRESHOLD)//if(var>IS_VALID_THRESHOLD)// if (abs( (best_val - average)/ average )> IS_VALID_THRESHOLD )
				{
					isValidImg.at(x,y) = 255;//abs((best_val - average)/ average );//1;
				}
			}
		}

		//Boundary
		for (int y = 0; y < iHeight; y++) {
			labelImg.at(0,y) = labelImg.at(1,y);
			labelImg.at(iWidth-1,y) = labelImg.at(iWidth-2,y);
		}
		for (int x = 0; x < iWidth; x++) {
			labelImg.at(x,0) = labelImg.at(x,1);
			labelImg.at(x,iHeight-1) = labelImg.at(x,iHeight-2);
		}

	}





void CDBP::Dmsg(float* s1, float* s2, float* s3, float* s4, float* dst, int offset, float w, float trunc) {
	float val;
	////added by dy
	//float * dst_old = new float[VALUES];
	//for (int i=0;i<VALUES;i++)
	//{
	//	dst_old[i] = dst[i];
	//}

	////added by dy end

	// offset: dstOffset - srcOffset
	// aggregate and find min
	float minimum = INF;

	for(int value = 0; value <VALUES; value++){
		dst[value] = INF;
	}
	for (int value = 0; value < VALUES; value++) {
		//message y->x: h(y) + |x-y+offset|
		//h(y), y = value, dstOffset + x = srcOffset + y
		float hVal = s1[value] + s2[value] + s3[value] + s4[value];		

		//Calculating minimum should be here! Corrected by Xiangli Kong.
		if (hVal < minimum)
			minimum = hVal;

		int xVal = min(VALUES-1, max(0,value - offset));
		hVal += abs(xVal - (value-offset));
		dst[xVal] = min(dst[xVal],hVal);

		//if (hVal < minimum)
		//	minimum = hVal;
	}

	// dt
	Dt(dst,w);

	// truncate 
	minimum += w * trunc;
	for (int value = 0; value < VALUES; value++)
		if (minimum < dst[value])
			dst[value] = minimum;

	// normalize 
	val = 0;
	for (int value = 0; value < VALUES; value++) 
		dst[value] /= 4;///val += dst[value];

	///val /= VALUES;
	///for (int value = 0; value < VALUES; value++) 
		///dst[value] -= val;
	//added by dy
	//static FILE * fout=fopen("dst.txt","w+");

	//for (int value = 0; value < VALUES; value++) 
	//	dst[value] = (dst[value]+ dst_old[value])*0.25;
	//if (dst[5]>0)
	//	fprintf(fout,"%f\n ",dst[5]);
	////fclose(fout);


	//delete [] dst_old;dst_old = NULL;
	///*for (int value = 0; value < VALUES; value++) 
	//	dst_old[value] = dst[value];*/

	////added by dy end
	}


void CDBP::DBp_CP(ZCubeFloatImage& uI, ZCubeFloatImage& dI, ZCubeFloatImage& rI, ZCubeFloatImage& lI, 
							  ZCubeFloatImage& dataCost, ZIntImage& offsetImg, ZFloatImage& wImg, ZFloatImage& truncImg)
{
int width = dataCost.GetWidth();
int height = dataCost.GetHeight();

int ITER = m_iMaxIter;


	if(m_iThreadCount <=1){
		for (int t = 0; t < ITER; t++) {
			std::cout << "iter " << t << "\n";

			for (int y = 1; y < height-1; y++) {
				for (int x = ((y+t) % 2) + 1; x < width-1; x+=2) {

					//message (x,y) -> (x,y-1)
					Dmsg(&uI.at(x,y+1),&lI.at( x+1, y),&rI.at( x-1, y),
						&dataCost.at(x, y), &uI.at(x, y), offsetImg.at(x,y-1)-offsetImg.at(x,y), wImg.at(x,y,2), truncImg.at(x,y,2));

					//(x,y) -> (x,y+1)
					Dmsg(&dI.at(x, y-1),&lI.at(x+1, y),&rI.at(x-1, y),
						&dataCost.at(x, y), &dI.at(x, y), offsetImg.at(x,y+1)-offsetImg.at(x,y), wImg.at(x,y,3), truncImg.at(x,y,3));

					//(x,y) -> (x+1,y)
					Dmsg(&uI.at(x, y+1),&dI.at(x, y-1),&rI.at(x-1, y),
						&dataCost.at(x, y), &rI.at(x, y), offsetImg.at(x+1,y)-offsetImg.at(x,y), wImg.at(x,y,1), truncImg.at(x,y,1));

					//(x,y) -> (x-1,y)
					Dmsg(&uI.at(x, y+1),&dI.at(x, y-1),&lI.at(x+1, y),
						&dataCost.at(x, y), &lI.at(x, y), offsetImg.at(x-1,y)-offsetImg.at(x,y), wImg.at(x,y,0), truncImg.at(x,y,0));

				}
			}
		}
	}
	else{
		/*std::cout << "Multi-thread BP:\n" <<endl;
		for (int t = 0; t < ITER; t++) {
			std::cout << "iter " << t << "\n";

			ParallelManager pm;
			pm.SetSize(m_iThreadCount);

			for (int y = 1; y < height-1; y++) {
				CExWRegularGridBPWorkUnit* pWorkUnit = new CExWRegularGridBPWorkUnit(y,width,t,uI,dI,rI,lI,dataCost,offsetImg,wImg,truncImg);

				pm.EnQueue(pWorkUnit);
			}

			pm.Run();
		}*/
	}
}




}


