#include "ExWRegularGridBP.h"

namespace NumericalAlgorithm{

class CDBP : public  CExWRegularGridBP
{

private:
	double IS_VALID_THRESHOLD;
	int iterCnt;
public:
	CDBP(double threshold = 0):IS_VALID_THRESHOLD(threshold),iterCnt(0){}
	void Dmsg(float* s1, float* s2, float* s3, float* s4, float* dst, int offset, float w, float trunc);
	void DBp_CP(ZCubeFloatImage& uI, ZCubeFloatImage& dI, ZCubeFloatImage& rI, ZCubeFloatImage& lI, 
		ZCubeFloatImage& dataCost, ZIntImage& offsetImg, ZFloatImage& wImg, ZFloatImage& truncImg);
	void DSolve(ZCubeFloatImage& dataCost, ZIntImage& labelImg, ZIntImage& offsetImg, ZIntImage& isValidImg, ZFloatImage& wImg, ZFloatImage& truncImg);
};

}