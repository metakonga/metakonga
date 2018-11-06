#ifndef matrix4x4_H
#define matrix4x4_H

#include <cmath>
#include <cassert>

#define MAT4x4 16

namespace algebra
{
	template< typename T >
	class matrix4x4
	{
	public:
		typedef T			value_t;
		typedef size_t		index_t;

	public:
		matrix4x4()
			: a00(0)
			, a01(0)
			, a02(0)
			, a03(0)
			, a10(0)
			, a11(0)
			, a12(0)
			, a13(0)
			, a20(0)
			, a21(0)
			, a22(0)
			, a23(0)
			, a30(0)
			, a31(0)
			, a32(0)
			, a33(0)
			, size(16)
			, diag(false)
		{}
		matrix4x4(T _a00, T _a01, T _a02, T _a03
			, T _a10, T _a11, T _a12, T _a13
			, T _a20, T _a21, T _a22, T _a23
			, T _a30, T _a31, T _a32, T _a33)
			: a00(_a00), a01(_a01), a02(_a02), a03(_a03)
			, a10(_a10), a11(_a11), a12(_a12), a13(_a13)
			, a20(_a20), a21(_a21), a22(_a22), a23(_a23)
			, a30(_a30), a31(_a31), a32(_a32), a33(_a33)
			, size(16), diag(false)
		{}

		~matrix4x4()
		{}

	public:
		T& operator()(index_t r, index_t c)
		{
			return *((&a00) + r * 4 + c);
		}

		T& operator()(index_t id)
		{
			return *((&a00) + id);
		}

		void setDiagonal(T v)
		{
			a00 = v; a11 = v; a22 = v; a33 = v;
		}

		matrix4x4& operator=(matrix4x4 const& m)
		{
			a00 = m.a00; a01 = m.a01; a02 = m.a02; a03 = m.a03;
			a10 = m.a10; a11 = m.a11; a12 = m.a12; a13 = m.a13;
			a20 = m.a20; a21 = m.a21; a22 = m.a22; a23 = m.a23;
			a30 = m.a30; a31 = m.a31; a32 = m.a32; a33 = m.a33;
			return *this;
		}

		matrix4x4& operator=(T v)
		{
			a00 = v; a01 = v; a02 = v; a03 = v;
			a10 = v; a11 = v; a12 = v; a13 = v;
			a20 = v; a21 = v; a22 = v; a23 = v;
			a30 = v; a31 = v; a32 = v; a33 = v;
			return *this;
		}

		void operator+=(matrix4x4 const& m)
		{
			a00 += m.a00; a01 += m.a01; a02 += m.a02; a03 += m.a03;
			a10 += m.a10; a11 += m.a11; a12 += m.a12; a13 += m.a13;
			a20 += m.a20; a21 += m.a21; a22 += m.a22; a23 += m.a23;
			a30 += m.a30; a31 += m.a31; a32 += m.a32; a33 += m.a33;
		}

		matrix4x4 operator+(matrix4x4 const& m)
		{
			return matrix4x4(
				a00 + m.a00, a01 + m.a01, a02 + m.a02, a03 + m.a03,
				a10 + m.a10, a11 + m.a11, a12 + m.a12, a13 + m.a13,
				a20 + m.a20, a21 + m.a21, a22 + m.a22, a23 + m.a23,
				a30 + m.a30, a31 + m.a31, a32 + m.a32, a33 + m.a33
				);
		}

		matrix4x4 operator-(matrix4x4 const& m)
		{
			return matrix4x4(
				a00 - m.a00, a01 - m.a01, a02 - m.a02, a03 - m.a03,
				a10 - m.a10, a11 - m.a11, a12 - m.a12, a13 - m.a13,
				a20 - m.a20, a21 - m.a21, a22 - m.a22, a23 - m.a23,
				a30 - m.a30, a31 - m.a31, a32 - m.a32, a33 - m.a33
				);
		}

		matrix4x4 operator-()
		{
			return matrix4x4(
				-a00, -a01, -a02, -a03,
				-a10, -a11, -a12, -a13,
				-a20, -a21, -a22, -a23,
				-a30, -a31, -a32, -a33
				);
		}

		void set(T _a00, T _a01, T _a02, T _a03
			, T _a10, T _a11, T _a12, T _a13
			, T _a20, T _a21, T _a22, T _a23
			, T _a30, T _a31, T _a32, T _a33)
		{
			a00 = _a00; a01 = _a01; a02 = _a02; a03 = _a03;
			a10 = _a10; a11 = _a11; a12 = _a12; a13 = _a13;
			a20 = _a20; a21 = _a21; a22 = _a22; a23 = _a23;
			a30 = _a30; a31 = _a31; a32 = _a32; a33 = _a33;
		}

	public:
		T a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
		size_t size;
		bool diag;
	};

	template <typename T2, typename T>
	inline matrix4x4<T> operator*(T2 s, matrix4x4<T>& m)
	{
		return matrix4x4<T>(
			m(0)*s, m(1)*s, m(2)*s, m(3)*s,
			m(4)*s, m(5)*s, m(6)*s, m(7)*s,
			m(8)*s, m(9)*s, m(10)*s, m(11)*s,
			m(12)*s, m(13)*s, m(14)*s, m(15)*s
			);
	}

}
#endif