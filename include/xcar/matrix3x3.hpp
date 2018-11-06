#ifndef MATRIX3X3_H
#define MATRIX3X3_H

#include <iostream>
#include <iomanip>

#include <cmath>
#include <cassert>

#define MAT3X3 9

template< typename base_type >
class matrix3x3
{
public:
	typedef unsigned int uint;

public:
	matrix3x3()
		: a00(0)
		, a01(0)
		, a02(0)
		, a10(0)
		, a11(0)
		, a12(0)
		, a20(0)
		, a21(0)
		, a22(0)
		, size(9)
		, diag(false)
	{}
	matrix3x3(base_type _a00, base_type _a01, base_type _a02
		, base_type _a10, base_type _a11, base_type _a12
		, base_type _a20, base_type _a21, base_type _a22)
		: a00(_a00), a01(_a01), a02(_a02)
		, a10(_a10), a11(_a11), a12(_a12)
		, a20(_a20), a21(_a21), a22(_a22)
		, size(9), diag(false)
	{}
	matrix3x3(base_type *data)
	{
		for(int i(0); i < 9; i++){
			*((&a00)+i) = data[i];
		}
		size=9;
		diag=false;
	}

	~matrix3x3()
	{}

public:
	base_type& operator()(unsigned r, unsigned c )
	{
		return *((&a00) + r*3+c);
	}

	base_type& operator()(unsigned id)
	{
		return *((&a00) + id);
	}

	void operator=(const base_type val)
	{
		a00=val; a01=val; a02=val;
		a10=val; a11=val; a12=val;
		a20=val; a21=val; a22=val;
	}

	matrix3x3& operator=(matrix3x3 const& m)
	{
		a00=m.a00; a01=m.a01; a02=m.a02;
		a10=m.a10; a11=m.a11; a12=m.a12;
		a20=m.a20; a21=m.a21; a22=m.a22;
		return *this;
	}

	matrix3x3 operator+ (matrix3x3 const& m)
	{
		return matrix3x3(a00+m.a00, a01+m.a01, a02+m.a02
			,a10+m.a10, a11+m.a11, a12+m.a12
			,a20+m.a20, a21+m.a21, a22+m.a22);
	}

	matrix3x3 operator-() const
	{
		return matrix3x3(
			-a00, -a01, -a02,
			-a10, -a11, -a12,
			-a20, -a21, -a22);
	}

	matrix3x3 t()
	{
		return matrix3x3(
			a00, a10, a20,
			a01, a11, a21,
			a02, a12, a22);
	}

public:
	void diagonal(base_type* diag)
	{
		a00 = diag[0]; a11 = diag[1]; a22 = diag[2];
	}

	void set(base_type _a00, base_type _a01, base_type _a02
		, base_type _a10, base_type _a11, base_type _a12
		, base_type _a20, base_type _a21, base_type _a22)
	{
		a00=_a00; a01=_a01; a02=_a02;
		a10=_a10; a11=_a11; a12=_a12;
		a20=_a20; a21=_a21; a22=_a22;
	}

	matrix3x3 inv()
	{
		base_type det=a00*a11 - a01*a10 - a00*a21 + a01*a20 + a10*a21 - a11*a20;
		matrix3x3 temp(
			(a11*a22 - a12*a21)/det,-(a01*a22 - a02*a21)/det, (a01*a12 - a02*a11)/det,
			-(a10*a22 - a12*a20)/det, (a00*a22 - a02*a20)/det,-(a00*a12 - a02*a10)/det,
			(a10*a21 - a11*a20)/det,-(a00*a21 - a01*a20)/det, (a00*a11 - a01*a10)/det
			);
		//*this = temp;
		return temp;
	}

public:
	base_type a00,a01,a02,a10,a11,a12,a20,a21,a22;
	size_t size;
	bool diag;
};

template< typename T >
std::ostream& operator<<(std::ostream& os, matrix3x3<T>& v)
{
	std::cout << std::endl;
//	std::ios::right;
	std::cout << v.a00 << "  " << v.a01 << "  " << v.a02 << std::endl 
		<< v.a10 << "  " << v.a11 << "  " << v.a12 << std::endl
		<< v.a20 << "  " << v.a21 << "  " << v.a22 << std::endl;
	std::cout << std::endl;
	return os;
}

// template <typename T2, typename T>
// inline matrix3x3<T> operator*( T2 s, matrix3x3<T>& m )  { return matrix3x3<T>( m(0)*s, m(1)*s, m(2)*s
// 																			  , m(3)*s, m(4)*s, m(5)*s
// 																			  , m(6)*s, m(7)*s, m(8)*s); }
template <typename T2, typename T>
inline matrix3x3<T> operator/( matrix3x3<T>& m, T2 s )  { return matrix3x3<T>( m(0)/s, m(1)/s, m(2)/s
																			  , m(3)/s, m(4)/s, m(5)/s
																			  , m(6)/s, m(7)/s, m(8)/s); }


#endif