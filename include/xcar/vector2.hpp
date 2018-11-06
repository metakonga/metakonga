#ifndef VECTOR2_H
#define VECTOR2_H

#include <iostream>

namespace algebra
{
	template < typename T >
	class vector2
	{
	public: 
		vector2() : x(0), y(0) {}
		vector2(T _x, T _y) : x(_x), y(_y) {}
		vector2(T val) : x(val), y(val) {}
		vector2(const vector2& vec2) : x(vec2.x), y(vec2.y) {}
		~vector2() {}

		vector2 operator- () const { return vector2(-x, -y); }
		vector2 operator+ (vector2& v2) const { return vector2(x + v2.x, y + v2.y); }
		vector2 operator+ (const vector2& v2) const { return vector2(x + v2.x, y + v2.y); }
		void operator+= (vector2& v2){ x+=v2.x; y+=v2.y; }
		vector2 operator- (vector2& v2) const { return vector2(x - v2.x, y - v2.y); }
		vector2 operator- (const vector2& v2) const { return vector2(x - v2.x, y - v2.y); }
		vector2 operator/ (T v) const { return vector2(x / v, y / v); }
		vector2 operator* (T v) const { return vector2(x * v, y * v); }



		inline vector2 rotate(T angle) const
		{
			return vector2(x * cos(angle) - y * sin(angle), y * cos(angle) + x * sin(angle));
		}

		inline vector2 normalize() const
		{
			return vector2(x,y) / length();
		}

		inline T length() const
		{
			return sqrt(x * x + y * y);
		}

		inline T dot(const vector2& v2) const 
		{
			return x * v2.x + y * v2.y;
		}

		inline T lengthSq() const
		{
			return x * x + y * y;
		}

	public:
		T x, y;
	};
	inline//template< typename T1, typename T2 >
		vector2<double> operator* (double sv, vector2<double> vv){
			return vector2<double>(sv * vv.x, sv * vv.y);
	}
	inline
		vector2<float> operator* (float sv, vector2<float> vv){
		return vector2<float>(sv * vv.x, sv * vv.y);
	}
// 	template < typename T >
// 	vector2<T> operator* (T sv, vector2<T>& vv){
// 		return vector2<T>(sv * vv.x, sv * vv.y);
// 	}
// 	template < typename T >
// 	vector2<T> operator* (vector2<T>& vv, T sv){
// 		return vector2<T>(sv * vv.x, sv * vv.y);
// 	}

	template< typename T >
	std::ostream& operator<<(std::ostream& os, vector2<T>& v)
	{
//		std::ios::right;
		os << v.x << " " << v.y;
		return os;
	}
}

#endif