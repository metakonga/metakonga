#ifndef MATRIX3X4_H
#define MATRIX3X4_H

#include <cmath>
#include <cassert>

#define MAT3X4 12

namespace algebra
{
	template< typename T >
	class matrix3x4
	{
	public:
		typedef T			value_t;
		typedef size_t		index_t;

	public:
		matrix3x4()
			: a00(0), a01(0), a02(0), a03(0)
			, a10(0), a11(0), a12(0), a13(0)
			, a20(0), a21(0), a22(0), a23(0)
			, size(12)
			, diag(false)
		{}
		matrix3x4(T _a00, T _a01, T _a02, T _a03
			, T _a10, T _a11, T _a12, T _a13
			, T _a20, T _a21, T _a22, T _a23)
			: a00(_a00), a01(_a01), a02(_a02), a03(_a03)
			, a10(_a10), a11(_a11), a12(_a12), a13(_a13)
			, a20(_a20), a21(_a21), a22(_a22), a23(_a23)
			, size(12), diag(false)
		{}

		~matrix3x4()
		{}

		//void B()

	public:
		T& operator()(index_t r, index_t c)
		{
			return *((&a00) + r*4+c);
		}

		T& operator()(index_t id)
		{
			return *((&a00) + id);
		}

		matrix3x4& operator=(matrix3x4 const& m)
		{
			a00=m.a00; a01=m.a01; a02=m.a02; a03=m.a03;
			a10=m.a10; a11=m.a11; a12=m.a12; a13=m.a13;
			a20=m.a20; a21=m.a21; a22=m.a22; a23=m.a23;
			return *this;
		}

		matrix3x4 operator+ (matrix3x4& m)
		{
			return matrix3x4(
				a00+m.a00,a01+m.a01,a02+m.a02,a03+m.a03,
				a10+m.a10,a11+m.a11,a12+m.a12,a13+m.a13,
				a20+m.a20,a21+m.a21,a22+m.a22,a23+m.a23);
		}

		matrix3x4 operator- (matrix3x4& m)
		{
			return matrix3x4(
				a00-m.a00,a01-m.a01,a02-m.a02,a03-m.a03,
				a10-m.a10,a11-m.a11,a12-m.a12,a13-m.a13,
				a20-m.a20,a21-m.a21,a22-m.a22,a23-m.a23);
		}

		matrix3x4 operator- ()
		{
			return matrix3x4(
				-a00, -a01, -a02, -a03,
				-a10, -a11, -a12, -a13,
				-a20, -a21, -a22, -a23);
		}

	public:
		T a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23;
		size_t size;
		bool diag;
	};

	template <typename T2, typename T>
	inline matrix3x4<T> operator*(T2 s, matrix3x4<T>& m)
	{
		return matrix3x4<T>(
			m(0)*s, m(1)*s, m(2)*s, m(3)*s,
			m(4)*s, m(5)*s, m(6)*s, m(7)*s,
			m(8)*s, m(9)*s, m(10)*s, m(11)*s);
	}

// 	template <typename T2, typename T>
// 	inline matrix3x4<T> operator*( T2 s, matrix3x4<T>& m )  
// 	{ 
// 		return matrix3x4<T>( 
// 			m(0,0)*s, m(0,1)*s, m(0,2)*s, m(0,3)*s,
// 			m(1,0)*s, m(1,1)*s, m(1,2)*s, m(1,3)*s,
// 			m(2,0)*s, m(2,1)*s, m(2,2)*s, m(2,3)*s
// 			); 
// 	}
}

#endif