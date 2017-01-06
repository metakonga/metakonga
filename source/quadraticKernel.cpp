#include "quadraticKernel.h"
#include "sphydrodynamics.h"

quadraticKernel::quadraticKernel(sphydrodynamics *_sph)
	: kernel(_sph)
{
	kernel_support = 2.5f;
	kernel_support_sq = kernel_support * kernel_support;
	if (sph->dimension() == DIM3){
		kernel_const = 1.f / (120.f * (float)M_PI * sph->smoothingKernel().h_inv_3);
	}
	else{
		kernel_const = 96.f / (1199.f * (float)M_PI * sph->smoothingKernel().h_sq);
	}
	kernel_grad_const = (-4.f) * kernel_const * sph->smoothingKernel().h_inv_sq;
}

quadraticKernel::~quadraticKernel()
{

}

float quadraticKernel::sphKernel(float QSq)
{
	float Q = sqrt(QSq);
	if (Q < 0.5f)
		return kernel_const * (pow(2.5f - Q, 4.f) - 5 * pow(1.5f - Q, 4.f) + 10 * pow(0.5f - Q, 4.f));
	else if (Q < 1.5f)
		return kernel_const * (pow(2.5f - Q, 4.f) - 5 * pow(1.5f - Q, 4.f));
	else if (Q < 2.5f)
		return kernel_const * pow(2.5f - Q, 4.f);

	return 0.0f;
}

vector3<float> quadraticKernel::sphKernelGrad(float QSq, VEC3F& distVec)
{
	float Q = sqrt(QSq);
	if (Q < 0.5f)
		return (kernel_grad_const / Q) * (pow(2.5f - Q, 3.f) - 5 * pow(1.5f - Q, 3.f) + 10 * pow(0.5f - Q, 3.f)) * distVec;
	else if (Q < 1.5f)
		return (kernel_grad_const / Q) * (pow(2.5f - Q, 3.f) - 5 * pow(1.5f - Q, 3.f)) * distVec;
	else if (Q < 2.5f)
		return (kernel_grad_const / Q) * pow(2.5f - Q, 3.f) * distVec;

	return 0.0f;
}