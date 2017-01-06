#include "cubic_spline_kernel.h"
#include "sphydrodynamics.h"
//#include <cmath>

using namespace parsph;

cubic_spline::cubic_spline(sphydrodynamics *_sph)
	: base_kernel(_sph)
{
	kernel_support = 2;
	kernel_support_sq = kernel_support * kernel_support;
	if(sph->Dimension()==DIM3){
		kernel_const = 1 / (M_PI * sph->Kernel().h_inv_3);
	}
	else{
		kernel_const = 10.0 / (7.0 * M_PI * sph->Kernel().h_sq);
	}
	kernel_grad_const = (-3.0/4.0) * kernel_const * sph->Kernel().h_inv;
}

cubic_spline::~cubic_spline()
{

}

double cubic_spline::sphKernel(double QSq)
{
	double Q = sqrt(QSq);
	if(0 <= Q  && Q < 1.0)
		return kernel_const * (1 - 1.5 * pow(Q, 2) + 0.75 * pow(Q,3));
	else if(1.0 <= Q && Q <= 2.0)
		return kernel_const * 0.25 * pow(2 - Q, 3);

	return 0.0;
}

vector3<double> cubic_spline::sphKernelGrad(double QSq, vector3<double>& distVec)
{
	double Q = sqrt(QSq);
	if(Q < 1.0)
		return kernel_grad_const * Q * (4.0 - 3.0 * Q) * (distVec / distVec.length());
	else {
		double dif = 2 - Q;
		return kernel_grad_const * dif * dif * (distVec / distVec.length());
	}
	
	return 0.0;
}