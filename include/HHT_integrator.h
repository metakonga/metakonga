#ifndef HHT_INTEGRATOR_H
#define HHT_INTEGRATOR_H

#include "Integrator.h"

namespace parSIM
{
	class HHT_integrator : public Integrator
	{
	public:
		HHT_integrator(Simulation* _sim);
		~HHT_integrator();

		virtual void integration();
		virtual void cu_integration();

		bool step;

	private:
		double alpha;
		double beta;
		double gamma;
		double dt2accp;
		double dt2accv;
		double dt2acc;
		double divalpha;
		double divbeta;
	};
}

#endif