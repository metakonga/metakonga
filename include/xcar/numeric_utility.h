#ifndef NUMERIC_UTILITY_H
#define NUMERIC_UTILITY_H

#include "algebraMath.h"

namespace numeric
{
	class utility
	{
	public:
		utility();
		~utility();
		static double signed_volume_of_triangle(VEC3D& v1, VEC3D& v2, VEC3D& v3);
		static VEC3D calculate_center_of_triangle(VEC3D& P, VEC3D& Q, VEC3D& R);
		static VEC3F calculate_center_of_triangle_f(VEC3F& P, VEC3F& Q, VEC3F& R);
		static double angle_coefficient(double d, double th);
		static double getMinValue(double v1, double v2, double v3);
		static double getMaxValue(double v1, double v2, double v3);
	};
}

#endif