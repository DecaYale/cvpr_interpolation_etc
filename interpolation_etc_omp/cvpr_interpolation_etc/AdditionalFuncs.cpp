
#include "ZImage.h"

void offsetGenerate(ZIntImage& dispMap,ZIntImage & offsetImg, int deviation)
{
	int width = dispMap.GetWidth();
	int height = dispMap.GetHeight();
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			int offset_t = dispMap.at(x,y) - deviation;
			offsetImg.at(x,y) = (offset_t) >=0 ? offset_t :0 ;
		}
		
	}
}