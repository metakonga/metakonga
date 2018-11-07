#include "numeric_utility.h"

using namespace numeric;

utility::utility(){}

utility::~utility(){}

double numeric::utility::signed_volume_of_triangle(VEC3D& v1, VEC3D& v2, VEC3D& v3)
{
	double v321 = v3.x*v2.y*v1.z;
	double v231 = v2.x*v3.y*v1.z;
	double v312 = v3.x*v1.y*v2.z;
	double v132 = v1.x*v3.y*v2.z;
	double v213 = v2.x*v1.y*v3.z;
	double v123 = v1.x*v2.y*v3.z;
	return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123);
}

VEC3D numeric::utility::calculate_center_of_triangle(VEC3D& P, VEC3D& Q, VEC3D& R)
{
	VEC3D V = Q - P;
	VEC3D W = R - P;
	VEC3D N = V.cross(W);
	N = N / N.length();
	VEC3D M1 = (Q + P) / 2;
	VEC3D M2 = (R + P) / 2;
	VEC3D D1 = N.cross(V);
	VEC3D D2 = N.cross(W);
	double t;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	}
	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	}
	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	{
		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	}
	
	return M1 + t * D1;
}

VEC3F numeric::utility::calculate_center_of_triangle_f(VEC3F& P, VEC3F& Q, VEC3F& R)
{
	VEC3F V = Q - P;
	VEC3F W = R - P;
	VEC3F N = V.cross(W);
	N = N / N.length();
	VEC3F M1 = (Q + P) / 2;
	VEC3F M2 = (R + P) / 2;
	VEC3F D1 = N.cross(V);
	VEC3F D2 = N.cross(W);
	float t;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	}
	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	}
	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	{
		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	}
	return M1 + t * D1;
}

double numeric::utility::angle_coefficient(double d, double th)
{
	unsigned int n = 0;
	unsigned int b = 0;
	unsigned int e = 1;
	unsigned int c = 0;
	double v = 0.0;
	if (th == 0.0)
		return th;
	while (1)
	{
		double _b = b * M_PI;
		double _e = e * M_PI;
		if (d > _b && d < _e)
		{
			if (!c)
				return th;
			n = 2 * (e / 2);
			unsigned int m = c % 2;
			if (m)
				v = n * M_PI - th;
			else
				v = n * M_PI + th;
			return v;
		}
		else
		{
			b++;
			e++;
			c++;
		}
	}
	return 0.0;
}

double numeric::utility::getMinValue(double v1, double v2, double v3)
{
	return v1 < v2 ? (v1 < v3 ? v1 : (v3 < v2 ? v3 : v2)) : (v2 < v3 ? v2 : v3);
}

double numeric::utility::getMaxValue(double v1, double v2, double v3)
{
	return v1 > v2 ? (v1 > v3 ? v1 : (v3 > v2 ? v3 : v2)) : (v2 > v3 ? v2 : v3);
}

// void numeric::utility::swap_column(MATD& lhs, unsigned int i, unsigned int j)
// {
// 
// }

void numeric::utility::coordinatePartioning(MATD& lhs, VECUI& pv)
{
	unsigned int nr = lhs.rows();
	for (unsigned int i = 0; i < nr; i++)
	{
		double mv = 0.0;
		unsigned int pv_c = 0.0;
		unsigned int k = 0;
		for (k = i; k < lhs.cols(); k++)
		{
			if (abs(lhs(i, k)) > abs(mv))
			{
				mv = lhs(i, k);
				pv_c = k;
			}
		}
		if (i != pv_c)
		{
			lhs.swap_column(i, pv_c);
			pv.swap(i, pv_c);
		}
			
		double inv_m = 1.0 / mv;
		lhs(i, i) *= inv_m;
		for (unsigned int jc = i + 1; jc < lhs.cols(); jc++)
		{
			double bc = inv_m * lhs(i, jc);
		//	double ec = lhs(jr, jc) + bc;
			lhs(i, jc) = bc;
			//lhs(jr, jc) = ec;
		}
		
		for (unsigned int jr = i + 1; jr < nr; jr++)
		{
			double m = -lhs(jr, i);
			lhs(jr, i) = 0;
			for (unsigned int jc = i + 1; jc < lhs.cols(); jc++)
			{
				//double bc = inv_m * lhs(i, jc);
				double ec = lhs(i, jc) * m;
				//lhs(i, jc) = bc;
				double vv = lhs(jr, jc) + ec;
				lhs(jr, jc) = vv;
			}
		}
	}
	//lhs.display();
}