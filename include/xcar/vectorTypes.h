#ifndef VECTORTYPES_H
#define VECTORTYPES_H

#include "matrix3x3.hpp"
#include "matrix4x3.hpp"
#include "matrix4x4.hpp"
#include "vector3.hpp"
#include "vector4.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "euler_parameter.hpp"

using namespace algebra;

typedef vector2<unsigned int>	VEC2UI;
typedef vector2<double>			VEC2D;
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
typedef matrix4x3<double>		MAT43D;
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

typedef struct{ double xx, xy, xz, yy, yz, zz; }symatrix;
typedef struct{ double s0, s1, s2, s3, s4, s5; }double6;

#endif