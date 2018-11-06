#ifndef VECTOR3_H
#define VECTOR3_H

#include "vector2.hpp"

namespace algebra
{
	template < typename T >
	class vector3
	{
	public: 
		vector3() : x(0), y(0), z(0) {}
		vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
		vector3(T val) : x(val), y(val), z(val) {}
		vector3(T *ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}
		vector3(const vector3& vec3) : x(vec3.x), y(vec3.y), z(vec3.z) {}
		vector3(const vector2<T>& vec2) : x(vec2.x), y(vec2.y), z(0) {}
		~vector3() {}

		template< typename TF>
		vector3<TF> To()
		{
			return vector3<TF>(static_cast<TF>(x), static_cast<TF>(y), static_cast<TF>(z));
		}

		void zeros() { x = 0; y = 0; z = 0; }
		T length() { return sqrt(x*x + y*y + z*z); }
		T dot() { return x*x + y*y + z*z; }
		template< typename T2 >
		T dot(vector3<T2>& v3) { return x*v3.x + y*v3.y + z*v3.z; }
		T dot(vector3& v3) { 
			return x*v3.x + y*v3.y + z*v3.z; 
		}
		T volume() { return x * y * z; }
		vector3 cross(vector3& v3) { return vector3(y*v3.z - z*v3.y, z*v3.x - x*v3.z, x*v3.y - y*v3.x); }
		vector3 rotate2(T angle) const
		{
			return vector3(x * cos(angle) - y * sin(angle), y * cos(angle) + x * sin(angle), 0);
		}

		//vector3 operator* (T val){ return vector3(val*x, val*y, val*z); }
		vector3 operator+ (const vector3& v3) const{ return vector3(x + v3.x, y + v3.y, z + v3.z); }
		template< typename T2 >
		vector3 operator- (vector3& v3) { return vector3(x - v3.x, y - v3.y, z - v3.z); }
		vector3 operator- (const vector3& v3) const 
		{ 
			return vector3(x - v3.x, y - v3.y, z - v3.z); 
		}
		void operator-= (vector3& v3){ x -= v3.x; y -= v3.y; z -= v3.z; }
		void operator+= (vector3& v3){ x += v3.x; y += v3.y; z += v3.z; }
		vector3 operator- () { return vector3(-x, -y, -z); }
		vector3 operator/ (T val) { return vector3(x/val, y/val, z/val); }
		bool operator== (vector3& v3) { return (x == v3.x) && (y == v3.y) && (z == v3.z); }
		void operator*= (T val) { x *= val; y *= val; z *= val; }
		void operator=(T val) { x = val; y = val; z = val; }
		bool operator<=(vector3& v3) { return (x <= v3.x&&y <= v3.y&&z <= v3.z); }
		bool operator>=(vector3& v3) { return (x >= v3.x&&y >= v3.y&&z >= v3.z); }
		T& operator() (unsigned id){ return *((&x) + id); }
		inline vector3 normalize() { return vector3(x,y,z) / length(); }
		vector2<T> toVector2() const { return vector2<T>(x, y); }
		T SignX() { return T(x <= 0 ? (x == 0 ? 0 : -1) : 1); }
		T SignY() { return T(y <= 0 ? (y == 0 ? 0 : -1) : 1); }
		T SignZ() { return T(z <= 0 ? (z == 0 ? 0 : -1) : 1); }
		vector3 Sign3() { return vector3(SignX(), SignY(), SignZ()); }

	public:
		T x, y, z;
	};

	inline//template< typename T1, typename T2 >
	vector3<double> operator* (double sv, vector3<double> vv){
		return vector3<double>(sv * vv.x, sv * vv.y, sv * vv.z);
	}
	inline
	vector3<float> operator* (float sv, vector3<float> vv){
		return vector3<float>(sv * vv.x, sv * vv.y, sv * vv.z);
	}
	inline
		vector3<int> operator* (int sv, vector3<int> vv){
		return vector3<int>(sv * vv.x, sv * vv.y, sv * vv.z);
	}
// 	vector3<double> operator* (double sv, vector3<double>& vv){
// 		return vector3<double>(sv * vv.x, sv * vv.y, sv * vv.z);
// 	}
// 	template < typename T >
// 	vector3<T> operator* (vector3<T>& vv, T sv){
// 		return vector3<T>(sv * vv.x, sv * vv.y, sv * vv.z);
// 	}

	template< typename T >
	std::ostream& operator<<(std::ostream& os, vector3<T>& v)
	{
//		std::ios::right;
		os << v.x << " " << v.y << " " << v.z;
		return os;
	}
}

#endif