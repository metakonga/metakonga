#ifndef PARABOLA_PREDICTOR_H
#define PARABOLA_PREDICTOR_H

#include "algebraMath.h"
//#include <fstream>

namespace utility
{
	class parabola_predictor
	{
	public:
		parabola_predictor();
		~parabola_predictor();

	public:
		bool apply(unsigned int it);

		double& getTimeStep() { return dt; }

		void init(double* _data, int _dataSize);

	private:
		VEC3I idx;
		VEC3D xp;
		VEC3D yp;
		VEC3D coeff;
		MAT33D A;

		VECD* data3;

		int dataSize;

		double* data;
		double dt;
	};
}

#endif