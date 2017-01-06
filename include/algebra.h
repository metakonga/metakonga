#ifndef ALGEBRA_H
#define ALGEBRA_H

#include "algebra/sparse_matrix.hpp"
#include "algebra/matrix.hpp"
#include "algebra/matrix3x3.hpp"
#include "algebra/matrix3x4.hpp"
#include "algebra/matrix4x4.hpp"
#include "algebra/vector.hpp"
#include "algebra/vector2.hpp"
#include "algebra/vector3.hpp"
#include "algebra/vector4.hpp"
#include "algebra/euler_parameter.hpp"

#define PI 3.14159265358979323846

typedef unsigned char byte;
typedef unsigned short word;

using namespace algebra;

//inline float frand() { return rand() / (float)RAND_MAX; }
inline float ffrand() { return rand() / (float)RAND_MAX; }

template <typename base_type>
inline base_type frand(base_type maxv) { return rand() / ((base_type)RAND_MAX / maxv); }

#define sign(a) a <= 0 ? (a == 0 ? 0 : -1) : 1

#define POINTER(a) &a(0)
#define POINTER3(a) &(a.x)
#define POINTER4(a) &(a.x)

typedef algebra::vector<int> vectori;
typedef algebra::vector2<int> vector2i;
typedef algebra::vector3<int> vector3i;
typedef algebra::vector4<int> vector4i;

typedef algebra::vector<size_t> vectoru;
typedef algebra::vector2<size_t> vector2u;
typedef algebra::vector3<size_t> vector3u;
typedef algebra::vector4<size_t> vector4u;

typedef algebra::vector<float> vectorf;
typedef algebra::vector2<float> vector2f;
typedef algebra::vector3<float> vector3f;
typedef algebra::vector4<float> vector4f;

typedef algebra::vector<double> vectord;
typedef algebra::vector2<double> vector2d;
typedef algebra::vector3<double> vector3d;
typedef algebra::vector4<double> vector4d;

typedef algebra::vector<vector2i> vvector2i;
typedef algebra::vector<vector3i> vvector3i;
typedef algebra::vector<vector4i> vvector4i;

typedef algebra::vector<vector2u> vvector2u;
typedef algebra::vector<vector3u> vvector3u;
typedef algebra::vector<vector4u> vvector4u;

typedef algebra::vector<vector2f> vvector2f;
typedef algebra::vector<vector3f> vvector3f;
typedef algebra::vector<vector4f> vvector4f;

typedef algebra::vector<vector2d> vvector2d;
typedef algebra::vector<vector3d> vvector3d;
typedef algebra::vector<vector4d> vvector4d;

typedef struct { double q, w, e, r; } scalar4;
typedef struct { double q, w, e, r, a, s, d, f; } scalar8;

inline scalar8 make_sym_tensor8(double A, double B, double C, double D, double E, double F) { scalar8 s8 = {A, B, C, D, E, F, 0, 0}; return s8; }
inline scalar4 make_sym_tensor4(double A, double B, double C) { scalar4 s4 = {A, B, C, 0}; return s4; }

inline scalar8 operator/ (scalar8 s, double div)
{
	double prod = 1 / div;
	scalar8 s8 = { s.q * prod, s.w * prod, s.e * prod, s.r * prod, s.a * prod, s.s * prod, s.d * prod, s.f * prod };
	return s8;
}

inline scalar4 operator/ (scalar4 s, double div)
{
	double prod = 1 / div;
	scalar4 s4 = { s.q * prod, s.w * prod, s.e * prod, s.r * prod };
	return s4;
}

// inline void MakeTransformationMatrix(double* A, double* ep)
// {
// 	A[0]=2*(ep[0]*ep[0]+ep[1]*ep[1]-0.5);	A[1]=2*(ep[1]*ep[2]-ep[0]*ep[3]);		A[2]=2*(ep[1]*ep[3]+ep[0]*ep[2]);
// 	A[3]=2*(ep[1]*ep[2]+ep[0]*ep[3]);		A[4]=2*(ep[0]*ep[0]+ep[2]*ep[2]-0.5);	A[5]=2*(ep[2]*ep[3]-ep[0]*ep[1]);
// 	A[6]=2*(ep[1]*ep[3]-ep[0]*ep[2]);		A[7]=2*(ep[2]*ep[3]+ep[0]*ep[1]);		A[8]=2*(ep[0]*ep[0]+ep[3]*ep[3]-0.5);
// }
// 
// 
// inline vector3<double> operator*(matrix3x3<double>& m, vector3<double>& v)
// {
// 	return vector3<double>(m.a00 * v.x + m.a01 * v.y + m.a02 * v.z,
// 						   m.a10 * v.x + m.a11 * v.y + m.a12 * v.z,
// 						   m.a20 * v.x + m.a21 * v.y + m.a22 * v.z);
// }
// 
// inline 
// 	matrix3x4<double> G(euler_parameter<double>& e)
// {
// 	return matrix3x4<double>(
// 		-e.e1,  e.e0,  e.e3, -e.e2,
// 		-e.e2, -e.e3,  e.e0,  e.e1,
// 		-e.e3,  e.e2, -e.e1,  e.e0);
// }
// 
// inline 
// 	matrix3x4<double> L(euler_parameter<double>& e)
// {
// 	return matrix3x4<double>(
// 		-e.e1,  e.e0, -e.e3,  e.e2,
// 		-e.e2,  e.e3,  e.e0, -e.e1,
// 		-e.e3, -e.e2,  e.e1,  e.e0);
// }
// 
// inline
// 	vector4d transpose(vector3d& v3, matrix3x4<double>& m3x4)
// {
// 	return vector4d(
// 		v3.x * m3x4.a00 + v3.y * m3x4.a10 + v3.z * m3x4.a20,
// 		v3.x * m3x4.a01 + v3.y * m3x4.a11 + v3.z * m3x4.a21,
// 		v3.x * m3x4.a02 + v3.y * m3x4.a12 + v3.z * m3x4.a22,
// 		v3.x * m3x4.a03 + v3.y * m3x4.a13 + v3.z * m3x4.a23
// 		);
// }
// 
// inline 
// 	vector4d transpose(vector4d& v4, matrix4x4<double>& m4x4)
// {
// 	return vector4d(
// 		v4.w*m4x4.a00 + v4.x*m4x4.a10 + v4.y*m4x4.a20 + v4.z*m4x4.a30,
// 		v4.w*m4x4.a01 + v4.x*m4x4.a11 + v4.y*m4x4.a21 + v4.z*m4x4.a31,
// 		v4.w*m4x4.a02 + v4.x*m4x4.a12 + v4.y*m4x4.a22 + v4.z*m4x4.a32,
// 		v4.w*m4x4.a03 + v4.x*m4x4.a13 + v4.y*m4x4.a23 + v4.z*m4x4.a33
// 		);
// }
// 
// inline
// 	matrix4x4<double> transpose(const matrix3x4<double>& m4x3, matrix3x4<double>& m3x4)
// {
// 	return matrix4x4<double>(
// 		m4x3.a00 * m3x4.a00 + m4x3.a10 * m3x4.a10 + m4x3.a20 * m3x4.a20, m4x3.a00 * m3x4.a01 + m4x3.a10 * m3x4.a11 + m4x3.a20 * m3x4.a21, m4x3.a00 * m3x4.a02 + m4x3.a10 * m3x4.a12 + m4x3.a20 * m3x4.a22, m4x3.a00 * m3x4.a03 + m4x3.a10 * m3x4.a13 + m4x3.a20 * m3x4.a23,
// 		m4x3.a01 * m3x4.a00 + m4x3.a11 * m3x4.a10 + m4x3.a21 * m3x4.a20, m4x3.a01 * m3x4.a01 + m4x3.a11 * m3x4.a11 + m4x3.a21 * m3x4.a21, m4x3.a01 * m3x4.a02 + m4x3.a11 * m3x4.a12 + m4x3.a21 * m3x4.a22, m4x3.a01 * m3x4.a03 + m4x3.a11 * m3x4.a13 + m4x3.a21 * m3x4.a23,
// 		m4x3.a02 * m3x4.a00 + m4x3.a12 * m3x4.a10 + m4x3.a22 * m3x4.a20, m4x3.a02 * m3x4.a01 + m4x3.a12 * m3x4.a11 + m4x3.a22 * m3x4.a21, m4x3.a02 * m3x4.a02 + m4x3.a12 * m3x4.a12 + m4x3.a22 * m3x4.a22, m4x3.a02 * m3x4.a03 + m4x3.a12 * m3x4.a13 + m4x3.a22 * m3x4.a23,
// 		m4x3.a03 * m3x4.a00 + m4x3.a13 * m3x4.a10 + m4x3.a23 * m3x4.a20, m4x3.a03 * m3x4.a01 + m4x3.a13 * m3x4.a11 + m4x3.a23 * m3x4.a21, m4x3.a03 * m3x4.a02 + m4x3.a13 * m3x4.a12 + m4x3.a23 * m3x4.a22, m4x3.a03 * m3x4.a03 + m4x3.a13 * m3x4.a13 + m4x3.a23 * m3x4.a23
// 		);
// }
// 
// inline
// 	vector4d transpose(const matrix3x4<double>& m4x3, vector3d& v3)
// {
// 	return vector4d(
// 		m4x3.a00*v3.x + m4x3.a10*v3.y + m4x3.a20*v3.z,
// 		m4x3.a01*v3.x + m4x3.a11*v3.y + m4x3.a21*v3.z,
// 		m4x3.a02*v3.x + m4x3.a12*v3.y + m4x3.a22*v3.z,
// 		m4x3.a03*v3.x + m4x3.a13*v3.y + m4x3.a23*v3.z
// 		);
// }
// 
// inline matrix3x4<double> operator*(const matrix3x3<double>& a, const matrix3x4<double>& b)
// {
// 	return matrix3x4<double>(
// 		a.a00*b.a00 + a.a01*b.a10 + a.a02*b.a20, a.a00*b.a01 + a.a01*b.a11 + a.a02*b.a21, a.a00*b.a02 + a.a01*b.a12 + a.a02*b.a22, a.a00*b.a03 + a.a01*b.a13 + a.a02*b.a23,
// 		a.a10*b.a00 + a.a11*b.a10 + a.a12*b.a20, a.a10*b.a01 + a.a11*b.a11 + a.a12*b.a21, a.a10*b.a02 + a.a11*b.a12 + a.a12*b.a22, a.a10*b.a03 + a.a11*b.a13 + a.a12*b.a23,
// 		a.a20*b.a00 + a.a21*b.a10 + a.a22*b.a20, a.a20*b.a01 + a.a21*b.a11 + a.a22*b.a21, a.a20*b.a02 + a.a21*b.a12 + a.a22*b.a22, a.a20*b.a03 + a.a21*b.a13 + a.a22*b.a23
// 		);
// }
// 
// inline vector3<double> operator*(const matrix3x4<double>& m3x4, euler_parameter<double>& v4)
// {
// 	return vector3<double>(
// 		m3x4.a00*v4.e0 + m3x4.a01*v4.e1 + m3x4.a02*v4.e2 + m3x4.a03*v4.e3,
// 		m3x4.a10*v4.e0 + m3x4.a11*v4.e1 + m3x4.a12*v4.e2 + m3x4.a13*v4.e3,
// 		m3x4.a20*v4.e0 + m3x4.a21*v4.e1 + m3x4.a22*v4.e2 + m3x4.a23*v4.e3
// 		);
// }
// 
// inline
// 	matrix4x4<double> opMiner(vector3d& v)
// {
// 	return matrix4x4<double>(
// 		0, -v.x, -v.y, -v.z,
// 		v.x,    0,  v.z, -v.y,
// 		v.y, -v.z,    0,  v.x,
// 		v.z,  v.y, -v.x,    0);
// }
// 
// inline 
// 	matrix3x4<double> B(euler_parameter<double>& e, vector3<double>& s)
// {
// 	return matrix3x4<double>(
// 		2*(2*s.x*e.e0+e.e2*s.z-e.e3*s.y), 2*(2*s.x*e.e1+e.e3*s.z+e.e2*s.y), 2*(e.e1*s.y+e.e0*s.z),				  2*(e.e1*s.z-e.e0*s.y),
// 		2*(2*s.y*e.e0-e.e1*s.z+e.e3*s.x), 2*(s.y*e.e1-e.e0*s.z),				2*(2*s.y*e.e2+e.e3*s.z+e.e1*s.x), 2*(e.e2*s.z+e.e0*s.x),
// 		2*(2*s.z*e.e0-e.e2*s.x+e.e1*s.y), 2*(s.z*e.e1+e.e0*s.y),				2*(e.e3*s.y-e.e0*s.x),			  2*(2*s.z*e.e3+e.e2*s.y+e.e1*s.x)
// 		);
// }
// 
// 
// inline double GaussianNormalDistribution(double x)
// {
// 	return (x > 0 ? 1 : -1) * (1 / sqrt(2 * PI))*exp(-x*x / (2));
// }

#endif