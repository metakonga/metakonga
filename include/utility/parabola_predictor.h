#ifndef PARABOLA_PREDICTOR_H
#define PARABOLA_PREDICTOR_H

#include "algebra.h"
//#include <fstream>

namespace utility
{
	class parabola_predictor
	{
	public:
		parabola_predictor();
		~parabola_predictor();

	public:
		bool apply(int it);

		double& getTimeStep() { return dt; }

		void init(double* _data, int _dataSize);

	private:
		vector3i idx;
		vector3d xp;
		vector3d yp;
		vector3d coeff;
		matrix3x3<double> A;

		algebra::vector<double>* data3;

		int dataSize;

		double* data;
		double dt;
	};
}

#endif