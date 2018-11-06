#ifndef VECTOR4_H
#define VECTOR4_H

#include "vector3.hpp"

namespace algebra
{
	template < typename T >
	class vector4
	{
	public: 
		vector4() : x(0), y(0), z(0), w(0) {}
		vector4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
		vector4(T val) : x(val), y(val), z(val), w(val) {}
		vector4(vector3<T>& v3, T _w) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}
		vector4(const vector4& vec4) : x(vec4.x), y(vec4.y), z(vec4.z), w(vec4.w) {}
		~vector4() {}

		T dot()
		{
			return x*x + y*y + z*z + w*w;
		}

		template< typename TF>
		vector4<TF> To()
		{
			return vector4<TF>(static_cast<TF>(x), static_cast<TF>(y), static_cast<TF>(z), static_cast<TF>(w));
		}

		vector4 operator+ (vector4& v4){
			return vector4(x + v4.x, y + v4.y, z + v4.z, w + v4.w);
		}
		vector4 operator- (vector4& v4){
			return vector4(x - v4.x, y - v4.y, z - v4.z, w - v4.w);
		}
// 		template<typename T1>
// 		vector4 operator* (T1 val){
// 			return vector4(val * x, val * y, val * z, val * w)
// 		}
		T& operator() (unsigned id){
			//assert(id > 3 && "Error - T vector4::operator() (int id > 3)");
			return *((&x) + id);
		}
		void operator+= (vector4& v4){ x += v4.x; y += v4.y; z += v4.z; w += v4.w; }
		void operator+= (vector3<T>& v3){ x += v3.x; y += v3.y; z += v3.z; }
		vector3<T> toVector3() { return vector3<T>(x, y, z); }
	//	void plusDataFromVec3(vector3<T>& val) { x += val.x; y += val.y; z += val.z; }

	public:
		T x, y, z, w;
	};

	inline//template< typename T1, typename T2 >
		vector4<double> operator* (double sv, vector4<double> vv){
		return vector4<double>(sv * vv.x, sv * vv.y, sv * vv.z, sv * vv.w);
	}
	inline
		vector4<float> operator* (float sv, vector4<float> vv){
		return vector4<float>(sv * vv.x, sv * vv.y, sv * vv.z, sv * vv.w);
	}
	inline
		vector4<int> operator* (int sv, vector4<int> vv){
		return vector4<int>(sv * vv.x, sv * vv.y, sv * vv.z, sv * vv.w);
	}
	template< typename T >
	std::ostream& operator<<(std::ostream& os, vector4<T>& v)
	{
//		std::ios::right;
		os << v.x << " " << v.y << " " << v.z << " " << v.w;
		return os;
	}
}

#endif