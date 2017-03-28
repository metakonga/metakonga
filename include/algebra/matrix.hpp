#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>

#include "algebra/matrix3x3.hpp"
#include "algebra/matrix3x4.hpp"
#include "algebra/matrix4x4.hpp"

//#include "algebra/typeDef.h"

using namespace std;

namespace algebra
{
	template< typename T >
	class matrix
	{
	private:
		typedef T	element_t;

	public:
		matrix() : nsize(0), nrow(0), ncol(0), elements(0)
		{

		}
		matrix(size_t r, size_t c) : nsize(r*c), nrow(r), ncol(c), elements(0)
		{
			elements = new element_t[nsize];
		}
		matrix(size_t r, size_t c, element_t init) : nsize(r*c), nrow(r), ncol(c), elements(0)
		{
			elements = new element_t[nsize];
			size_t i = nsize;
			while(i--) elements[i] = init;
		}
		matrix(matrix<T>& _mat) : nsize(0), nrow(0), ncol(0), elements(0)
		{
			nsize = _mat.sizes();
			nrow = _mat.rows();
			ncol = _mat.cols();
			elements = new element_t[nsize];
			for(size_t i(0); i < nrow; i++){
				for(size_t j(0); j < ncol; j++){
					elements[j * nrow + i] = _mat(i,j);
				}
			}
		}

		~matrix()
		{
			if(elements) delete [] elements; elements = 0;
		}

	public:
		void alloc(int r, int c)
		{
			if(elements)
			{
				return;
			}
			nsize = r*c;
			nrow = r;
			ncol = c;
			elements = new element_t[nsize];
		}

		void resize(size_t r, size_t c)
		{
			if(!elements) 
			{
				elements = new element_t[r*c];
				this->nsize = r*c;
				this->nrow = r;
				this->ncol = c;
				return;
			}
			delete [] elements; elements=0;
			elements = new element_t[r*c];
			this->nsize = r*c;
			this->nrow = r;
			this->ncol = c;
		}

		void in2(size_t r1, size_t c1, size_t r2, size_t c2, double val)
		{
			elements[c1*nrow+r1]+=val;
			elements[c2*nrow+r2]+=-val;
		}

		void insert(size_t sr, size_t sc, matrix3x4<double>& m3x4)
		{
			unsigned cnt=0;
			for(size_t r(sr); r<sr+3; r++)
				for(size_t c(sc); c<sc+4; c++)
					elements[c * nrow + r] = *((&m3x4.a11 + cnt++));
		}

		void zeros()
		{
			for(unsigned i(0); i < nsize; i++)
				elements[i] = 0;
		}

		void display()
		{
			fstream fs;
			fs.open("C:/C++/mat.txt", ios::out);
			cout << endl;
			ios::right;
			for(size_t i(0); i < nrow; i++){
				for(size_t j(0); j < ncol; j++){
					fs/*cout*/ << setprecision(18) << setw(26) << elements[j * nrow + i];
				}
				fs << endl;
			}
			fs << endl;
			fs.close();
		}

		void plus(size_t sr, size_t sc, element_t* ptr, int type)
		{
			unsigned cnt = 0;
			switch(type){
			case MAT3X4:
				for(size_t r(sr); r<sr+3; r++)
					for(size_t c(sc); c<sc+4; c++)
						elements[c * nrow + r] += *(ptr + cnt++);//*((&m3x4.a00 + cnt++));
				break;
			case MAT4x3:
				for(size_t r(sr); r<sr+3; r++)
					for(size_t c(sc); c<sc+4; c++)
						elements[r * nrow + c] += *(ptr + cnt++);//*((&m3x4.a00 + cnt++));
				break;
			case MAT4x4:
				for(size_t r(sr); r<sr+4; r++)
					for(size_t c(sc); c<sc+4; c++)
						elements[c * nrow + r] += *(ptr + cnt++);
				break;
			}
		}

		element_t* getDataPointer() { return elements; }

		size_t& sizes() { return nsize; }
		size_t& rows() { return nrow; }
		size_t& cols() { return ncol; }


	public:
		// operator
		element_t& operator() (const size_t r, const size_t c) const { return elements[c * nrow + r]; }
		void operator= (const element_t val) const
		{
			size_t i = nsize;
			while(i--) elements[i] = val;
		}

		void operator= (matrix& _mat)
		{
// 			nsize = _mat.sizes();
// 			nrow = _mat.rows();
// 			ncol = _mat.cols();
			//elements = new element_t[nsize];
			for(size_t i(0); i < nrow; i++){
				for(size_t j(0); j < ncol; j++){
					elements[j * nrow + i] = _mat(i,j);
				}
			} 
		}
		

	private:
		size_t nsize;
		size_t nrow;
		size_t ncol;

		element_t* elements;
	};

// 	template <typename T, typename T2>
// 	matrix<T> operator*(T2 const& s, matrix<T>& m)
// 	{
// 		matrix<T> out(m.rows(), m.cols());
// 		for (size_t i(0); i < m.rows(); i++){
// 			for (size_t j(0); j < m.cols(); j++){
// 				out(i, j) = s*m(i, j);
// 			}
// 		}
// 		return out;
// 	}
}

#endif