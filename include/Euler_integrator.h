#ifndef EULER_INTEGRATOR_H
#define EULER_INTEGRATOR_H

#include "Integrator.h"
#include "particles.h"

namespace parSIM
{
	class Euler_integrator : public Integrator
	{
	public:
		Euler_integrator(Simulation* _sim);
		~Euler_integrator();

		//template< typename T >
		virtual void integration();
		virtual void cu_integration();
	};
}

#endif