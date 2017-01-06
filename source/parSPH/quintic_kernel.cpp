#include "quintic_kernel.h"
#include "sphydrodynamics.h"
//#include <cmath>

using namespace parsph;

quintic::quintic(sphydrodynamics *_sph)
	: base_kernel(_sph)
{
	kernel_support = 3;
	kernel_support_sq = kernel_support * kernel_support;
	if(sph->Dimension()==DIM3){
		kernel_const = M_1_PI / 120 * sph->Kernel().h_inv_3;
	}
	else{
		kernel_const = 7 * M_1_PI / 478 * sph->Kernel().h_inv_sq;
	}
	kernel_grad_const = -5 * kernel_const * sph->Kernel().h_inv_sq;
}

quintic::~quintic()
{

}

double quintic::sphKernel(double QSq)
{
	double Q = sqrt(QSq);
	if(Q < 1.0)
		return kernel_const * (pow(3-Q, 5) - 6*pow(2-Q, 5) + 15 * pow(1-Q, 5));
	else if(Q < 2.0)
		return kernel_const * (pow(3-Q, 5) - 6*pow(2-Q, 5));
	else
		return kernel_const * pow(3-Q, 5);
}

vector3<double> quintic::sphKernelGrad(double QSq, vector3<double>& distVec)
{
	double Q = sqrt(QSq);
	if(Q < 1.0)
		return (kernel_grad_const / Q * (pow(3-Q, 4) - 6*pow(2-Q, 4) + 15*pow(1-Q, 4))) * distVec;
	else if(Q < 2.0)
		return (kernel_grad_const / Q * (pow(3-Q, 4) - 6*pow(2-Q, 4))) * distVec;
	else
		return (kernel_grad_const / Q * pow(3-Q, 4)) * distVec;
}