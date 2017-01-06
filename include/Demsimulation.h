#ifndef DEMSIMULATION_H
#define DEMSIMULATION_H

#include "Simulation.h"

namespace parSIM
{
	class force;
	class cell_grid;

	class Demsimulation : public Simulation
	{
	public:
		Demsimulation(std::string name);
		virtual ~Demsimulation();
		
		void setIntegration(integrator_type itype) { integrator = itype; }
		
		bool initialize();
		void Integration();
		void cu_integrator_binding_data(double* p, double *v, double* a, double* omg, double* aph, double* f, double* m);
		void cpu_run();
		void gpu_run();
		virtual bool initialize_simulation();
		virtual bool RunSim();

		integrator_type integrator;
		Integrator* itor[NUM_INTEGRATOR];
	};
}

#endif