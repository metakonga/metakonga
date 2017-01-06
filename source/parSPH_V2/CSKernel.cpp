#include "CSKernel.h"
#include "sphydrodynamics.h"

CSKernel::CSKernel(sphydrodynamics *_sph)
	: kernel(_sph)
{
	kernel_support = 2;
	kernel_support_sq = kernel_support * kernel_support;
	if (sph->dimension() == DIM3){
		kernel_const = 1 / ((float)M_PI * sph->smoothingKernel().h_inv_3);
	}
	else{
		kernel_const = 10.0f / (7.0f * (float)M_PI * sph->smoothingKernel().h_sq);
	}
	kernel_grad_const = (-3.0f / 4.0f) * kernel_const * sph->smoothingKernel().h_inv_sq;
}

CSKernel::~CSKernel()
{

}

float CSKernel::sphKernel(float QSq)
{
	float Q = sqrt(QSq);
	if (0 <= Q  && Q <= 1.0f)
		return kernel_const * (1.f - 1.5f * QSq + 0.75f * QSq * Q);
	else if (1.0f <= Q && Q <= 2.0f)
		return kernel_const * 0.25f * pow(2.f - Q, 3.f);

	return 0.0f;
}

vector3<float> CSKernel::sphKernelGrad(float QSq, VEC3F& distVec)
{
	float Q = sqrt(QSq);
	if (Q <= 1.0f)
		return kernel_grad_const/* * Q*/ * (4.0f - 3.0f * Q) * (distVec /*/ distVec.length()*/);
	else {
		float dif = 2.f - Q;
		return kernel_grad_const * dif * dif * (distVec / Q/*/ distVec.length()*/);
	}

	return 0.0f;
}