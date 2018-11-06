#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <iostream>
#include <iomanip>

#ifndef MAT3X4
#define MAT3X4 12
#endif
#ifndef VEC4
#define VEC4 4
#endif
#ifndef VEC3_4 
#define VEC3_4 34
#endif

namespace xdyn
{
template< typename T >
class sparse_matrix
{
public:
	sparse_matrix()
		: _nnz(0)
		, max_nnz(0)
		, nrow(0)
		, ncol(0)
		, ridx(0)
		, cidx(0)
		, count(0)
		, value(0)
	{

	}

	sparse_matrix(int _max_nnz, int nRow, int nCol)
		: _nnz(0)
		, max_nnz(_max_nnz)
		, nrow(nRow)
		, ncol(nCol)
		, ridx(0)
		, cidx(0)
		, count(0)
		, value(0)
	{
		ridx = new int[max_nnz];
		cidx = new int[max_nnz];
		value = new T[max_nnz];
	}

	sparse_matrix(int r, int c)
		: nrow(r)
		, ncol(c)
		, max_nnz(r*c)
		, _nnz(0)
		, count(0)
		, ridx(0)
		, cidx(0)
		, value(0)
	{
		ridx = new int[max_nnz];
		cidx = new int[max_nnz];
		value = new T[max_nnz];
	}
	~sparse_matrix()
	{
		if(ridx) delete [] ridx; ridx = 0;
		if(cidx) delete [] cidx; cidx = 0;
		if(value) delete [] value; value = 0;
	}

	void alloc(int _max_nnz, int nRow, int nCol)
	{
		if (value)
			return;
		_nnz = 0;
		max_nnz = _max_nnz;
		nrow = nRow;
		ncol = nCol;
		ridx = 0;
		cidx = 0;
		count = 0;
		value = 0;
		ridx = new int[max_nnz];
		cidx = new int[max_nnz];
		value = new T[max_nnz];
	}

	int& nnz() { return _nnz; } 
	int& rows() { return nrow; }
	int& cols() { return ncol; }
	void zeroCount() { count = 0; _nnz = 0; }

	void resize(int nr, int nc)
	{
		nrow = nr; ncol = nc;
		max_nnz = nr * nc;
		if(!ridx) ridx = new int[max_nnz];
		if(!cidx) cidx = new int[max_nnz];
		if(!value) value = new T[max_nnz];
	}

	void extraction(int sr, int sc, T* ptr, int type)
	{
		unsigned cnt=0;
		T data=0;
		switch(type)
		{
		case MAT3X4:
			for(int i(0); i < 3; i++){
				for(int j(0); j < 4; j++){
					data = *(ptr+(cnt++));	
					if(data) (*this)(sr + i,sc + j) = data;
				}
			}
			break;
		case VEC4:
			for(unsigned i(0); i < 4; i++){
				data = *(ptr+(cnt++));
				if(data) (*this)(sr, sc + i) = data;
			}
			break;
		}

		_nnz = count;
	}

	void extraction(int sr, int sc, T* ptr1, T* ptr2, int type)
	{
		unsigned cnt=0;
		unsigned c=sc;
		T data = 0;
		switch(type)
		{
		case VEC3_4:
			for(; c < sc+3; c++){
				data = *(ptr1+(cnt++));
				if(data) (*this)(sr, c) = data;
			}
			cnt = 0;
			for(; c < sc+7; c++){
				data = *(ptr2+(cnt++));
				if(data) (*this)(sr, c) = data;
			}
			break;				
		}
		_nnz = count;
	}

public:
	T& operator()(const int r, const int c)
	{
		ridx[count] = r;
		cidx[count] = c;
		_nnz++;
		return value[count++];
	}
	void operator()(const int r, const int c, const T val)
	{
		if(val)
		{
			ridx[count] = r;
			cidx[count] = c;
			value[count++] = val;
		}
	}
private:
	int _nnz; 
	int count;
	int max_nnz;

	int nrow, ncol;

public:
	int* ridx;
	int* cidx;
	T *value;
};

template< typename T >
std::ostream& operator<<(std::ostream& os, sparse_matrix<T>& sm)
{
//	std::ios::right;
	std::setprecision(15);
	for(int i(0); i < sm.nnz(); i++)
	{
		os << "[" << i << "]" << " ";
		os << std::setw(5) << sm.ridx[i] << std::setw(5) << sm.cidx[i] << std::setw(28) << sm.value[i] << std::endl; 
	}
	return os;
}
}



#endif