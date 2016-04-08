#ifndef Nonlocal_H
#define Nonlocal_H
#include "tree_filter.h"
//#include"DataCost.h"
#include "Nonlocal/Nonlocal_basic.h"
#include <stdlib.h>
//#include"Block.h"

#include "cxcore.h"

union uname
{
	 int limit_data;
	unsigned char buffer[4];
};

class Nonlocal
{
private:
	//
public:
// 	static Nonlocal *GetInstance(){
// 		static Nonlocal Instance;
// 		return &Instance;
// 	}
	Nonlocal();
    ~Nonlocal();
    void clean();
	int init(int h,int w,int nr_plane,double sigma_range=0.1);
   void CostAggregation(bool refinement);
	void filter(float**image_filtered,float**image,bool compute_weight=true);
private:
	tree_filter m_tf;
	int	m_h,m_w,m_nr_plane;
	unsigned char**m_disparity,***m_buf_u2;
	double m_table[256],m_sigma_range;
	unsigned char***m_left;
	float ****m_buf_d3;
	float ***m_cost_vol,***m_cost_vol_temp;

public:
	void stereo(const cv::Mat & imgL,cv::Mat & datacost,cv::Mat & labelimg, double sigma,bool rfm);//void stereo(CMDRVideoFrame* pCurrentFrame,double sigma,CBlock block,CDataCost & datacost,ZIntImage& labelimg,bool rfm);

	//void m_detectUnstable(CBlock& block);
	//void RefineDspBySegm( const CBlock &block, const CMeanShiftSeg &meanShiftSegm,  CMDRVideoFrame* currentFrame, ZIntImage& labelImg, CDataCost& DataCost );
	//double m_getCostData(int x,int y,int lev){return m_cost_vol[y][x][lev];};
	//void RefineDspByBP(float disck, const CBlock &block, CDataCost &DataCost, ZIntImage &labelImg, bool addEdgeInfo, ZIntImage* offsetImg = NULL );
	//void m_SetCostData(int x,int y,int lev,double val){ m_cost_vol[y][x][lev]=val;};

	//void GetDataCost(const CBlock &block, CDataCost& outDataCost, ZIntImage& labelImg, const std::vector<double> &dspV, ZIntImage *pOffsetImg);
	//void Run_Detect_Unstable( int start, int end );
};
#endif