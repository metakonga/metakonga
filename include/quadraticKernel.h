#ifndef QUADRATICKERNEL_H
#define QUADRATICKERNEL_H

#include "kernel.h"

class quadraticKernel : public kernel
{
public:
	quadraticKernel(sphydrodynamics *_sph);
	virtual ~quadraticKernel();

	virtual float sphKernel(float QSq);
	virtual VEC3F sphKernelGrad(float QSq, VEC3F& distVec);
};

#endif