#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "Simulation.h"
#include "algebra.h"

using namespace algebra;

namespace parSIM
{
	class Integrator
	{
	public:
		Integrator(Simulation *_sim);
		virtual ~Integrator();

		virtual void integration() = 0;
		virtual void cu_integration() = 0;
		void setDt(double Dt) { dt = Dt; }
		void setNp(unsigned int _np) { np = _np; }
		void binding_data(vector4<double> *p, vector4<double> *v, vector4<double> *a, vector4<double> *omg, vector4<double> *aph, vector3<double> *f, vector3<double> *m);
		void cu_binding_data(double* p, double *v, double* a, double* omg, double* aph, double* f, double* m);

	protected:
		double dt;
		unsigned int np;
		vector4<double> *pos;
		vector4<double> *vel;
		vector4<double> *acc;
		vector4<double> *omega;
		vector4<double> *alpha;

		vector3<double>* force;
		vector3<double>* moment;

		double *d_pos;
		double *d_vel;
		double *d_acc;
		double *d_omega;
		double *d_alpha;

		double *d_force;
		double *d_moment;

		Simulation *sim;
	};
}

#endif