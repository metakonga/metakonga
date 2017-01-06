#ifndef EULER_PARAMETER_HPP
#define EULER_PARAMETER_HPP

#include "vector3.hpp"
#include "matrix3x3.hpp"
#include "matrix3x4.hpp"

namespace algebra
{
	template<typename T>
	class euler_parameter
	{
	public:
		euler_parameter() : e0(0), e1(0), e2(0), e3(0) {}
		euler_parameter(T _e0, T _e1, T _e2, T _e3) : e0(_e0), e1(_e1), e2(_e2), e3(_e3) {}
		euler_parameter(const euler_parameter& ep) : e0(ep.e0), e1(ep.e1), e2(ep.e2), e3(ep.e3) {}
		~euler_parameter(){}

		euler_parameter operator+(euler_parameter& v4){
			return euler_parameter(e0 + v4.e0, e1 + v4.e1, e2 + v4.e2, e3 + v4.e3);
		}

		T dot(){
			return e0*e0 + e1*e1 + e2*e2 + e3*e3;
		}

		matrix3x4<T> G()
		{
			return matrix3x4<double>(
				-e1, e0, e3, -e2,
				-e2, -e3, e0, e1,
				-e3, e2, -e1, e0);
		}

		matrix3x4<T> L()
		{
			return matrix3x4<double>(
				-e1, e0, -e3, e2,
				-e2, e3, e0, -e1,
				-e3, -e2, e1, e0);
		}

		template< typename TF>
		euler_parameter<TF> To()
		{
			return euler_parameter<TF>(static_cast<TF>(e0), static_cast<TF>(e1), static_cast<TF>(e2), static_cast<TF>(e3));
		}

		T& operator() (unsigned id){
			//assert(id > 3 && "Error - T vector4::operator() (int id > 3)");
			return *((&e0) + id);
		}

		T* Pointer() { return &e0; }

		void setFromEuler(T xi, T th, T ap){
			//e0 = cos(0.5 * th) * cos(0.5 * (xi + ap));
			e1 = sin(0.5 * th) * cos(0.5 * (xi - ap));
			e2 = sin(0.5 * th) * sin(0.5 * (xi - ap));
			e3 = cos(0.5 * th) * sin(0.5 * (xi + ap));
			e0 = sqrt(1.0 - (e1 * e1 + e2 * e2 + e3 * e3));
			//T len = _e0 * _e0 + e1 * e1 + e2 * e2 + e3 * e3;// dot();
		}

		void normalize(){
			T d = dot();
			e0 /= d; e1 /= d; e2 /= d; e3 /= d;
		}

		vector3<T> toAngularVelocity(euler_parameter<T>& ep)
		{
			return 2 * vector3<T>(
				-ep.e1 * e0 + ep.e0 * e1 - ep.e3 * e2 + ep.e2 * e3,
				-ep.e2 * e0 + ep.e3 * e1 + ep.e0 * e2 - ep.e1 * e3,
				-ep.e3 * e0 - ep.e2 * e1 + ep.e1 * e2 + ep.e0 * e3);
		}

		matrix3x3<T> A()
		{
			matrix3x3<T> TM;
			TM.a00 = 2 * (e0*e0 + e1*e1 - 0.5);	TM.a01 = 2 * (e1*e2 - e0*e3);		TM.a02 = 2 * (e1*e3 + e0*e2);
			TM.a10 = 2 * (e1*e2 + e0*e3);		TM.a11 = 2 * (e0*e0 + e2*e2 - 0.5);	TM.a12 = 2 * (e2*e3 - e0*e1);
			TM.a20 = 2 * (e1*e3 - e0*e2);		TM.a21 = 2 * (e2*e3 + e0*e1);		TM.a22 = 2 * (e0*e0 + e3*e3 - 0.5);
			return TM;
		}

		T e0, e1, e2, e3;
	};

	template<typename T2>
	euler_parameter<T2> operator*(int val, euler_parameter<T2>& ep){
		return euler_parameter<T2>(val * ep.e0, val * ep.e1, val * ep.e2, val * ep.e3);
	}

	template<typename T2>
	euler_parameter<T2> operator*(double val, euler_parameter<T2>& ep){
		return euler_parameter<T2>(val * ep.e0, val * ep.e1, val * ep.e2, val * ep.e3);
	}

	template<typename T2>
	euler_parameter<T2> operator*(unsigned int val, euler_parameter<T2>& ep){
		return euler_parameter<T2>(val * ep.e0, val * ep.e1, val * ep.e2, val * ep.e3);
	}
}

#endif