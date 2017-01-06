#ifndef ROCK_FORCE_H
#define ROCK_FORCE_H

#include "algebra.h"
#include "force.h"

namespace parSIM
{
	class cell_grid;
	class Simulation;

	class rock_force : public force
	{
	public:
		rock_force(Simulation *Sim);
		virtual ~rock_force();

		//double* Ks() { return ks; }

		bool calForce(unsigned int i, unsigned int j, particle *pars);//, double ir, double jr, vector3<double>& ip, vector3<double>& jp, vector3<double>& iv, vector3<double>& jv, vector3<double>& iw, vector3<double>& jw, vector3<double>& f, vector3<double>& m);

		virtual void initialize(particle* pars);
		virtual void collision(cell_grid *detector, particle *pars);
		virtual void cu_collision(cell_grid *detector, double* pos, double* vel, double* acc, double* omega, double* alpha, unsigned int cRun = 0);

		double mu;
		//double *kn, *ks;
		//vector3<double> *Fs;
	};
}

#endif