#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include <fstream>
#include <iomanip>

namespace algebra
{
	template<typename base_type>
	class vector
	{
	public:
		vector()
			: storage(0)
			, capacity(0)
			, count(0)
		{
			//storage = new base_type;
		}
		vector(unsigned size)
			: storage(0)
			, count(size)
			, capacity(size)
		{
			storage = new base_type[size];
			for(unsigned i(0); i < size; i++){
				storage[i] = 0;
			}
		}
		vector(unsigned size, base_type *ptr)
			: storage(0)
			, count(size)
			, capacity(size)
		{
			storage = new base_type[size];
			for(unsigned i(0); i < size; i++) 
				storage[i] = ptr[i];
		}
		vector(unsigned size, base_type init)
			: storage(0)
			, count(size)
			, capacity(size)
		{
			storage = new base_type[size];
			for(unsigned i(0); i < size; i++) storage[i]=init;
		}
		vector(const vector& _vec)
			: storage(0)
			, count(_vec.count)
			, capacity(_vec.capacity)
		{
			storage = new base_type[count];
			for(unsigned i(0); i < count; i++) storage[i]=_vec(i);
		}
		~vector()
		{
			if(storage) delete [] storage; storage=0;
			capacity=0;
			count=0;
		}
		void zeros() { for(unsigned i(0); i < capacity; i++) storage[i] = 0; }

		base_type&	operator() (const unsigned idx) const { return storage[idx]; }
		void		operator=  (const base_type v) { for(unsigned i(0); i < count; i++) storage[i] = v; }
		vector&	operator=  (const vector& vec) 
		{ 
			if(!storage)
			{
				capacity=vec.capacity;
				count=vec.count;
				storage=new base_type[capacity];
			}
			for(unsigned i(0); i < count; i++) 
				storage[i] = vec(i); 
			return *this; 
		}
		vector operator+ ( vector& v ) const { vector out(v.sizes()); for(unsigned i(0); i < v.sizes(); i++) out(i) = storage[i] + v(i); return out; }
		void	operator+=( vector& v ) const { for(unsigned i(0); i < v.sizes(); i++) storage[i]+=v(i); }
		void    operator-=( vector& v ) const { for(unsigned i(0); i < v.sizes(); i++) storage[i]-=v(i); }
		void    operator*=( base_type v) const { unsigned i = count; while(i--) storage[i] *= v; }
		vector operator* ( vector& v ) const { vector out(v.sizes()); for(unsigned i(0); i < v.sizes(); i++) out(i) = storage[i] * v(i); return out; }
		vector operator* (base_type v) { vector out(sizes()); for (unsigned i(0); i < sizes(); i++) out(i) = storage[i] * v; return out; }
		vector operator/ ( vector& v ) const { vector out(v.sizes()); for(unsigned i(0); i < v.sizes(); i++) out(i) = storage[i] / v(i); return out; }
//		vector operator/ (base_type v) const { vector out(sizes()); for (unsigned i(0); i < sizes(); i++) out(i) = storage[i] / v; return out; }
		vector operator- ( vector& v ) const { vector out(v.sizes()); for(unsigned i(0); i < v.sizes(); i++) out(i) = storage[i] - v(i); return out; }

		// 	void vec3(vector3<base_type>* data, unsigned i)
		// 	{ 
		// 		storage[i*3+0]=data->x; storage[i*3+1]=data->y; storage[i*3+2]=data->z;
		// 	}

		void getchar_ptr(const char* ptr)
		{
			if(storage) delete [] storage; storage = NULL;
			char pointer[256];
			sprintf_s(pointer, 256, "0x%s", ptr);
			storage = (base_type*) pointer;
		}

		void resize(unsigned _s/*, base_type init*/)
		{
			if(storage)
			{
				base_type* old=storage;
				storage = new base_type[_s];
				memcpy(storage, old, capacity*sizeof(base_type));
				if(old)
				{
					capacity ? delete [] old : delete old;
					old = 0;
				}
			}
			else
			{
				storage = new base_type[_s];
				//for(unsigned i(0); i < _s; i++) storage[i]=init;
			}
			capacity=_s;	
		}
		void alloc(unsigned _s/*, base_type init=0*/)
		{
			capacity=count=_s;
			storage = new base_type[_s];
			memset(storage, 0, sizeof(base_type) * _s);
		}

		void dealloc()
		{
			if(storage) delete [] storage; storage = NULL;
			capacity=count=0;
		}

		void CopyFromPtr(base_type* ptr)
		{
			for(unsigned i(0); i < count; i++) storage[i] = ptr[i];
		}

		void none() {}
		void countZero() { count = 0; }
		unsigned& fullSize() { return capacity; }
		base_type* at(unsigned id) { return storage + id; }
		unsigned& sizes() {return count; }
		base_type* begin() { return storage; }
		base_type* end() { return storage+count; }
		vector get_ref() { return *this; }
		base_type* get_ptr() { return storage; }
		void set_ptr(base_type* ptr) { storage = ptr; }
		void adjustment() { capacity = count; resize(count); }
		void push(base_type& _obj)
		{
			if(count==capacity)
			{
				resize(capacity*2+1);
			}
			if(storage)
			{
				storage[count++]=_obj;
			}
		}

		void toMinus()
		{
			for(unsigned i(0); i < count; i++)
				storage[i] = -storage[i];
		}

		void insert(size_t id, base_type &obj)
		{
			storage[id] = obj;
		}

		void insert(size_t s, size_t e, base_type* ptr)
		{
			unsigned cnt = 0;
			while(s <= e) 
				storage[s++] = *(ptr + cnt++);
		}

		void insert(int s, base_type* data, int num=0)
		{
			if(!num) num = this->sizes();
			for(int i(0); i < num; i++)
				storage[s++] = *(data + i);
		}

		void insert(int s, base_type* data1, base_type* data2, int n1, int n2)
		{
			int i=0;
			for(; i < n1; i++)
				storage[s++] = data1[i];
			for(i=0; i < n2; i++)
				storage[s++] = data2[i];
		}

		void plus(int s, base_type* data1, base_type* data2, int n1, int n2)
		{
			int i = 0;
			for (; i < n1; i++)
				storage[s++] += data1[i];
			for (i = 0; i < n2; i++)
				storage[s++] += data2[i];
		}

		void plus(unsigned s, unsigned e, base_type* ptr)
		{
			while(s <= e)
				storage[s++] += *(ptr++);
		}

		void plus(unsigned int s, base_type* ptr, int ndata = 0)
		{
			unsigned int limit = s+ndata;
			while(s < limit)
				storage[s++] += *(ptr++);
		}

		base_type norm()
		{
			base_type sum=0;
			for(unsigned i(0); i < count; i++) sum += storage[i]*storage[i];
			return sqrt(sum);
		}

		void initSequence()
		{
			for(unsigned i(0); i < count; i++)
				storage[i] = i;
		}

		void out2txt(char* name)
		{
			char filename[255] = {0, };
			sprintf_s(filename, 255, "%s%s", name, ".txt");
			std::fstream out;
			out.open(filename, std::ios::out);
			out << count << std::endl;
			for(unsigned i = 0; i < count; i++){
				out << storage[i] << std::endl;
			}
			std::cout << name << " is wrote." << std::endl;
		}

	private:
		base_type *storage;

	public:
		unsigned count;
		unsigned capacity;
	};
}

template< typename T >
std::ostream& operator<<(std::ostream& os, algebra::vector<T>& v)
{
	std::cout << std::endl;
////	std::ios::right;
	for(size_t i(0); i < v.sizes(); i++){
		os << " [" << i << "] " << std::setprecision(12) << v(i) << std::endl;
	}
	
	std::cout << std::endl;
	return os;
}

// template <typename T>
// algebra::vector<T> operator-( algebra::vector<T>& v) { algebra::vector<T> out(v.sizes()); for(size_t i(0); i < v.sizes(); i++) out(i) = -out(i); return out; }
// 
// template <typename T, typename T2>
// algebra::vector<T> operator*( algebra::vector<T>& v, T2 const& s ) { algebra::vector<T> out(v.sizes()); for(size_t i(0); i < v.sizes(); i++) out(i) = v(i)*s; return out; }
// 
// template <typename T, typename T2>
// algebra::vector<T> operator*( T2 const& s, algebra::vector<T>& v ) { algebra::vector<T> out(v.sizes()); for(size_t i(0); i < v.sizes(); i++) out(i) = v(i)*s; return out;}
// 
// template <typename T, typename T2>
// algebra::vector<T> operator/( algebra::vector<T>& v, T2 const& s ) {  algebra::vector<T> out(v.sizes());for(size_t i(0); i < v.sizes(); i++) out(i) = v(i)/s; return out; }

#endif