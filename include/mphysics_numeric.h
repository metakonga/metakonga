#ifndef MPHYSICS_NUMERIC_H
#define MPHYSICS_NUMERIC_H

#include "algebra/matrix3x3.hpp"
#include "algebra/matrix4x4.hpp"
#include "algebra/vector3.hpp"
#include "algebra/vector4.hpp"
#include "algebra/vector.hpp"
#include "algebra/matrix.hpp"
#include "algebra/sparse_matrix.hpp"
#include "algebra/euler_parameter.hpp"

using namespace algebra;

// #define min(a,b) a < b ? a : b
// #define max(a,b) a > b ? a : b

#define sign(a) a <= 0 ? (a == 0 ? 0 : -1) : 1

#define POINTER(a) &a(0)
#define POINTER3(a) &(a.x)
#define POINTER4(a) &(a.x)

inline float frand() { return rand() / (float)RAND_MAX; }

typedef vector3<float>			VEC3F;
typedef vector3<double>			VEC3D;
typedef vector3<int>			VEC3I;
typedef vector3<unsigned int>	VEC3UI;
typedef vector4<float>			VEC4F;
typedef vector4<double>			VEC4D;
typedef vector4<int>			VEC4I;
typedef vector4<unsigned int>	VEC4UI;
typedef algebra::vector<int>	VECI;
typedef algebra::vector<unsigned int> VECUI;
typedef algebra::vector<float>	VECF;
typedef algebra::vector<double>	VECD;
typedef matrix<float>			MATF;
typedef matrix<double>			MATD;
typedef matrix3x3<float>		MAT33F;
typedef matrix3x4<float>		MAT34F;
typedef matrix4x4<float>		MAT44F;
typedef matrix3x3<double>		MAT33D;
typedef matrix3x4<double>		MAT34D;
typedef matrix4x4<double>		MAT44D;
typedef euler_parameter<float>	EPF;
typedef euler_parameter<double>	EPD;
typedef xdyn::sparse_matrix<double>   SMATD;
typedef VECF*					VECF_PTR;
typedef VECD*					VECD_PTR;
typedef VECI*					VECI_PTR;
typedef VECUI*					VECUI_PTR;
typedef VEC3F*					VEC3F_PTR;
typedef VEC3D*					VEC3D_PTR;
typedef VEC4F*					VEC4F_PTR;
typedef VEC4D*					VEC4D_PTR;
typedef MAT33F*					MAT33F_PTR;
typedef MAT34F*					MAT34F_PTR;
typedef MAT44F*					MAT44F_PTR;
typedef MAT33D*					MAT33D_PTR;
typedef MAT34D*					MAT34D_PTR;
typedef MAT44D*					MAT44D_PTR;
typedef MATF*					MATF_PTR;
typedef MATD*					MATD_PTR;
typedef EPF*					EPF_PTR;
typedef EPD*					EPD_PTR;
typedef SMATD*					SMATD_PTR;

typedef vector4<float>			VEC4F;
typedef VEC4F*					VEC4F_PTR;

typedef struct
{
	VEC3D P;
	VEC3D Q;
	VEC3D R;
	VEC3D V;
	VEC3D W;
	VEC3D N;
}host_polygon_info;

typedef struct 
{
	VEC3D origin;
	VEC3D vel;
	VEC3D omega;
	EPD ep;
}host_polygon_mass_info;

inline
MAT44D transpose(const MAT34D& m4x3, MAT34D& m3x4)
{
	return MAT44D(
		m4x3.a00 * m3x4.a00 + m4x3.a10 * m3x4.a10 + m4x3.a20 * m3x4.a20, m4x3.a00 * m3x4.a01 + m4x3.a10 * m3x4.a11 + m4x3.a20 * m3x4.a21, m4x3.a00 * m3x4.a02 + m4x3.a10 * m3x4.a12 + m4x3.a20 * m3x4.a22, m4x3.a00 * m3x4.a03 + m4x3.a10 * m3x4.a13 + m4x3.a20 * m3x4.a23,
		m4x3.a01 * m3x4.a00 + m4x3.a11 * m3x4.a10 + m4x3.a21 * m3x4.a20, m4x3.a01 * m3x4.a01 + m4x3.a11 * m3x4.a11 + m4x3.a21 * m3x4.a21, m4x3.a01 * m3x4.a02 + m4x3.a11 * m3x4.a12 + m4x3.a21 * m3x4.a22, m4x3.a01 * m3x4.a03 + m4x3.a11 * m3x4.a13 + m4x3.a21 * m3x4.a23,
		m4x3.a02 * m3x4.a00 + m4x3.a12 * m3x4.a10 + m4x3.a22 * m3x4.a20, m4x3.a02 * m3x4.a01 + m4x3.a12 * m3x4.a11 + m4x3.a22 * m3x4.a21, m4x3.a02 * m3x4.a02 + m4x3.a12 * m3x4.a12 + m4x3.a22 * m3x4.a22, m4x3.a02 * m3x4.a03 + m4x3.a12 * m3x4.a13 + m4x3.a22 * m3x4.a23,
		m4x3.a03 * m3x4.a00 + m4x3.a13 * m3x4.a10 + m4x3.a23 * m3x4.a20, m4x3.a03 * m3x4.a01 + m4x3.a13 * m3x4.a11 + m4x3.a23 * m3x4.a21, m4x3.a03 * m3x4.a02 + m4x3.a13 * m3x4.a12 + m4x3.a23 * m3x4.a22, m4x3.a03 * m3x4.a03 + m4x3.a13 * m3x4.a13 + m4x3.a23 * m3x4.a23
		);
}

// inline 
// VEC4D transpose(const MAT34D& m4x3, VEC3D& v3)
// {
// 
// }

inline
VEC4D transpose(VEC3D& v3, MAT34D& m3x4)
{
	return VEC4D(
		v3.x * m3x4.a00 + v3.y * m3x4.a10 + v3.z * m3x4.a20,
		v3.x * m3x4.a01 + v3.y * m3x4.a11 + v3.z * m3x4.a21,
		v3.x * m3x4.a02 + v3.y * m3x4.a12 + v3.z * m3x4.a22,
		v3.x * m3x4.a03 + v3.y * m3x4.a13 + v3.z * m3x4.a23
		);
}

inline
VEC3F transpose(MAT33F& m, VEC3F& a)
{
	return VEC3F(
		a.x*m.a00 + a.y*m.a10 + a.z*m.a20,
		a.x*m.a01 + a.y*m.a11 + a.z*m.a21,
		a.x*m.a02 + a.y*m.a12 + a.z*m.a22);
}

inline
VEC3D transpose(VEC3D& a, MAT33D& m)
{
	return VEC3D(
		a.x*m.a00 + a.y*m.a10 + a.z*m.a20,
		a.x*m.a01 + a.y*m.a11 + a.z*m.a21,
		a.x*m.a02 + a.y*m.a12 + a.z*m.a22);
}

inline MAT34D operator*(const MAT33D& a, const MAT34D& b)
{
	return MAT34D(
		a.a00*b.a00 + a.a01*b.a10 + a.a02*b.a20, a.a00*b.a01 + a.a01*b.a11 + a.a02*b.a21, a.a00*b.a02 + a.a01*b.a12 + a.a02*b.a22, a.a00*b.a03 + a.a01*b.a13 + a.a02*b.a23,
		a.a10*b.a00 + a.a11*b.a10 + a.a12*b.a20, a.a10*b.a01 + a.a11*b.a11 + a.a12*b.a21, a.a10*b.a02 + a.a11*b.a12 + a.a12*b.a22, a.a10*b.a03 + a.a11*b.a13 + a.a12*b.a23,
		a.a20*b.a00 + a.a21*b.a10 + a.a22*b.a20, a.a20*b.a01 + a.a21*b.a11 + a.a22*b.a21, a.a20*b.a02 + a.a21*b.a12 + a.a22*b.a22, a.a20*b.a03 + a.a21*b.a13 + a.a22*b.a23
		);
}

inline
MAT34D B(EPD& e, VEC3D& s)
{
	return MAT34D(
		2 * (2 * s.x*e.e0 + e.e2*s.z - e.e3*s.y), 2 * (2 * s.x*e.e1 + e.e3*s.z + e.e2*s.y), 2 * (e.e1*s.y + e.e0*s.z), 2 * (e.e1*s.z - e.e0*s.y),
		2 * (2 * s.y*e.e0 - e.e1*s.z + e.e3*s.x), 2 * (s.y*e.e1 - e.e0*s.z), 2 * (2 * s.y*e.e2 + e.e3*s.z + e.e1*s.x), 2 * (e.e2*s.z + e.e0*s.x),
		2 * (2 * s.z*e.e0 - e.e2*s.x + e.e1*s.y), 2 * (s.z*e.e1 + e.e0*s.y), 2 * (e.e3*s.y - e.e0*s.x), 2 * (2 * s.z*e.e3 + e.e2*s.y + e.e1*s.x)
		);
}

inline VEC3D operator*(const MAT34D& m3x4, EPD& v4)
{
	return VEC3D(
		m3x4.a00*v4.e0 + m3x4.a01*v4.e1 + m3x4.a02*v4.e2 + m3x4.a03*v4.e3,
		m3x4.a10*v4.e0 + m3x4.a11*v4.e1 + m3x4.a12*v4.e2 + m3x4.a13*v4.e3,
		m3x4.a20*v4.e0 + m3x4.a21*v4.e1 + m3x4.a22*v4.e2 + m3x4.a23*v4.e3
		);
}

inline VEC3D operator*(MAT33D& m, VEC3D& v)
{
	return VEC3D(m.a00 * v.x + m.a01 * v.y + m.a02 * v.z,
		m.a10 * v.x + m.a11 * v.y + m.a12 * v.z,
		m.a20 * v.x + m.a21 * v.y + m.a22 * v.z);
}

inline VEC3F operator*(MAT33F& m, VEC3F& v)
{
	return VEC3F(m.a00 * v.x + m.a01 * v.y + m.a02 * v.z,
		m.a10 * v.x + m.a11 * v.y + m.a12 * v.z,
		m.a20 * v.x + m.a21 * v.y + m.a22 * v.z);
}

inline
MAT44D opMiner(VEC3D& v)
{
	return MAT44D(
		0, -v.x, -v.y, -v.z,
		v.x, 0, v.z, -v.y,
		v.y, -v.z, 0, v.x,
		v.z, v.y, -v.x, 0);
}

inline MAT33D tilde(VEC3D& v)
{
	return MAT33D(
		0, -v.z, v.y,
		v.z, 0, -v.x,
		-v.y, v.x, 0);
}

inline VEC3D ep2e(EPD& ep)
{
	//double d1 = ((ep.e1 * ep.e3) + (ep.e0 * ep.e2)) / -((ep.e2 * ep.e3) - (ep.e0 * ep.e1));
	double m33 = 2.0 * (ep.e0 * ep.e0 + ep.e3 * ep.e3 - 0.5);
	if (m33 < 1.0e-15)
	{
		m33 = 0.0;
	}
	return VEC3D(
		atan2(((ep.e1 * ep.e3) + (ep.e0 * ep.e2)), -((ep.e2 * ep.e3) - (ep.e0 * ep.e1))),
		atan2(sqrt(1 - m33 * m33), m33),
		atan2(((ep.e1 * ep.e3) - (ep.e0 * ep.e2)), ((ep.e2 * ep.e3) + ep.e0 * ep.e1)));
}

#endif