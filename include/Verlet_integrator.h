#ifndef VERLET_INTEGRATOR_H
#define VERLET_INTEGRATOR_H

#include "Integrator.h"

namespace parSIM
{

	class Verlet_integrator : public Integrator
	{
	public:
		Verlet_integrator(Simulation* _sim);
		~Verlet_integrator();

		virtual void integration();
		virtual void cu_integration();

		bool seq;
		Simulation *sim;
	};
}

#endif