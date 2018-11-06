#ifndef MATRIX4X3_H
#define MATRIX4X3_H

#include <cmath>
#include <cassert>

#define MAT4x3 21

namespace algebra
{
	template< typename T >
	class matrix4x3
	{
	public:
		typedef T			value_t;
		typedef size_t		index_t;

	public:
		matrix4x3()
			: a00(0), a01(0), a02(0)
			, a10(0), a11(0), a12(0)
			, a20(0), a21(0), a22(0)
			, a30(0), a31(0), a32(0)
			, size(12)
		{}
		matrix4x3(T _a00, T _a01, T _a02
			, T _a10, T _a11, T _a12
			, T _a20, T _a21, T _a22
			, T _a30, T _a31, T _a32)
			: a00(_a00), a01(_a01), a02(_a02)
			, a10(_a10), a11(_a11), a12(_a12)
			, a20(_a20), a21(_a21), a22(_a22)
			, a30(_a30), a31(_a31), a32(_a32)
			, size(12)
		{}

		~matrix4x3()
		{}

		//void B()

	public:
		T& operator()(index_t r, index_t c)
		{
			return *((&a00) + r * 3 + c);
		}

		T& operator()(index_t id)
		{
			return *((&a00) + id);
		}

		matrix4x3& operator=(matrix4x3 const& m)
		{
			a00 = m.a00; a01 = m.a01; a02 = m.a02;
			a10 = m.a10; a11 = m.a11; a12 = m.a12;
			a20 = m.a20; a21 = m.a21; a22 = m.a22;
			a30 = m.a30; a31 = m.a31; a32 = m.a32;
			return *this;
		}

		matrix4x3 operator+ (matrix4x3& m)
		{
			return matrix4x3(
				a00 + m.a00, a01 + m.a01, a02 + m.a02,
				a10 + m.a10, a11 + m.a11, a12 + m.a12,
				a20 + m.a20, a21 + m.a21, a22 + m.a22,
				a30 + m.a30, a31 + m.a31, a32 + m.a32);
		}

		matrix4x3 operator- (matrix4x3& m)
		{
			return matrix4x3(
				a00 - m.a00, a01 - m.a01, a02 - m.a02,
				a10 - m.a10, a11 - m.a11, a12 - m.a12,
				a20 - m.a20, a21 - m.a21, a22 - m.a22,
				a30 - m.a30, a31 - m.a31, a32 - m.a32);
		}

		matrix4x3 operator- ()
		{
			return matrix4x3(
				-a00, -a01, -a02,
				-a10, -a11, -a12,
				-a20, -a21, -a22,
				-a30, -a31, -a32);
		}

	public:
		T a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31, a32;
		size_t size;
		//bool diag;
	};

	template <typename T2, typename T>
	inline matrix4x3<T> operator*(T2 s, matrix4x3<T>& m)
	{
		return matrix4x3<T>(
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