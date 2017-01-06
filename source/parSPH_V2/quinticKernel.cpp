#include "quinticKernel.h"
#include "sphydrodynamics.h"

quinticKernel::quinticKernel(sphydrodynamics *_sph)
	: kernel(_sph)
{
	kernel_support = 3;
	kernel_support_sq = kernel_support * kernel_support;
	if (sph->dimension() == DIM3){
		kernel_const = 1.f / ( 120.f * (float)M_PI * sph->smoothingKernel().h_inv_3);
	}
	else{
		kernel_const = 7.f / (478.f * (float)M_PI * sph->smoothingKernel().h_sq);
	}
	kernel_grad_const = (-5.f) * kernel_const * sph->smoothingKernel().h_inv_sq;
}

quinticKernel::~quinticKernel()
{

}

float quinticKernel::sphKernel(float QSq)
{
	float Q = sqrt(QSq);
	if (Q < 1.0f)
		return kernel_const * (pow(3.f - Q, 5.f) - 6 * pow(2.f - Q, 5.f) + 15 * pow(1.f - Q, 5.f));
	else if (Q < 2.0f)
		return kernel_const * (pow(3.f - Q, 5.f) - 6 * pow(2.f - Q, 5.f));
	else if (Q < 3.f)
		return kernel_const * pow(3.f - Q, 5.f);

	return 0.0f;
}

vector3<float> quinticKernel::sphKernelGrad(float QSq, VEC3F& distVec)
{
	float Q = sqrt(QSq);
	if (Q < 1.0f)
		return (kernel_grad_const / Q) * (pow(3.f - Q, 4.f) - 6 * pow(2.f - Q, 4.f) + 15 * pow(1.0f - Q, 4.f)) * distVec;
	else if (Q < 2.f)
		return (kernel_grad_const / Q) * (pow(3.f - Q, 4.f) - 6 * pow(2.f - Q, 4.f)) * distVec;
	else if (Q < 3.f)
		return (kernel_grad_const / Q) * pow(3.f - Q, 4.f) * distVec;
	
	return 0.0f;
}