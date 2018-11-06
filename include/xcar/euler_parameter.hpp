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
		
		euler_parameter operator-(euler_parameter& v4){
			return euler_parameter(e0 - v4.e0, e1 - v4.e1, e2 - v4.e2, e3 - v4.e3);
		}

		euler_parameter operator-()
		{
			return euler_parameter(-e0, -e1, -e2, -e3);
		}

		T dot(){
			return e0*e0 + e1*e1 + e2*e2 + e3*e3;
		}

		T dot(euler_parameter& e) { return e0 * e.e0 + e1 * e.e1 + e2 * e.e2 + e3 * e.e3; }

		T length(){
			return sqrt(dot());
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
			e0 = cos(0.5 * th) * cos(0.5 * (xi + ap));
			e1 = sin(0.5 * th) * cos(0.5 * (xi - ap));
			e2 = sin(0.5 * th) * sin(0.5 * (xi - ap));
			e3 = cos(0.5 * th) * sin(0.5 * (xi + ap));
			double v = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
			if (v > 0) e0 = sqrt(1.0 - e1 * e1 + e2 * e2 + e3 * e3);
			//e0 = sqrt(1.0 - v);
			//T len = _e0 * _e0 + e1 * e1 + e2 * e2 + e3 * e3;// dot();
		}

		euler_parameter w2ev(vector3<T>& w)
		{
			return 0.5 * euler_parameter(
				-e1 * w.x - e2 * w.y - e3 * w.z,
				 e0 * w.x + e3 * w.y - e2 * w.z,
				-e3 * w.x + e0 * w.y + e1 * w.z,
				 e2 * w.x - e1 * w.y + e0 * w.z);
		}

		void normalize(){
			T d = sqrt(dot());
			e0 /= d; e1 /= d; e2 /= d; e3 /= d;
		}

		vector3<T> toAngularVelocity(euler_parameter<T>& ep)
		{
			return 2 * vector3<T>(
				-ep.e1 * e0 + ep.e0 * e1 - ep.e3 * e2 + ep.e2 * e3,
				-ep.e2 * e0 + ep.e3 * e1 + ep.e0 * e2 - ep.e1 * e3,
				-ep.e3 * e0 - ep.e2 * e1 + ep.e1 * e2 + ep.e0 * e3);
		}

		euler_parameter<T> toEP2GTV(vector3<T>& v)
		{
			return 2.0 * euler_parameter<T>(
				-e1 * v.x - e2 * v.y - e3 * v.z,
				e0 * v.x + e3 * v.y - e2 * v.z,
				-e3 * v.x + e0 * v.y + e1 * v.z,
				e2 * v.x - e1 * v.y + e0 * v.z);
		}

		vector3<T> toEP2GV(euler_parameter& v)
		{
			return 2.0 * vector3<T>(
				-e1 * v.e0 + e0 * v.e1 - e3 * v.e2 + e2 * v.e3,
				-e2 * v.e0 + e3 * v.e1 + e0 * v.e2 - e1 * v.e3, 
				-e3 * v.e0 - e2 * v.e1 + e1 * v.e2 + e0 * v.e3);
		}

		euler_parameter<T> toEulerParameter2dot(vector3<T>& a, vector3<T>& v)
		{
			T vp = 0.25 * v.dot() * (*this);
			euler_parameter ga = 0.5 * euler_parameter(-e1 * a.x - e2 * a.y - e3 * a.z,
				                          e0 * a.x + e3 * a.y - e2 * a.z,
										 -e3 * a.x + e0 * a.y + e1 * a.z,
										  e2 * a.x - e1 * a.y + e0 * a.z); 
			return ga + vp;
		}

		matrix3x3<T> A()
		{
			matrix3x3<T> TM;
			TM.a00 = 2 * (e0*e0 + e1*e1 - 0.5);	TM.a01 = 2 * (e1*e2 - e0*e3);		TM.a02 = 2 * (e1*e3 + e0*e2);
			TM.a10 = 2 * (e1*e2 + e0*e3);		TM.a11 = 2 * (e0*e0 + e2*e2 - 0.5);	TM.a12 = 2 * (e2*e3 - e0*e1);
			TM.a20 = 2 * (e1*e3 - e0*e2);		TM.a21 = 2 * (e2*e3 + e0*e1);		TM.a22 = 2 * (e0*e0 + e3*e3 - 0.5);
			return TM;
		}

		vector3<T> toLocal(vector3<T>& v)
		{
			vector3<T> tv;
			matrix3x3<T> TM;
			TM = A();
			tv.x = TM.a00*v.x + TM.a10*v.y + TM.a20*v.z;
			tv.y = TM.a01*v.x + TM.a11*v.y + TM.a21*v.z;
			tv.z = TM.a02*v.x + TM.a12*v.y + TM.a22*v.z;
			return tv;
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