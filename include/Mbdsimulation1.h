#ifndef MBDSIMULATION_H
#define MBDSIMULATION_H

#include "Simulation.h"



using namespace algebra;

namespace parSIM
{
	class pointmass;
	class mbd_force;

	class Mbdsimulation : public Simulation
	{
	public:
		Mbdsimulation(Simulation *sim = NULL);
		virtual ~Mbdsimulation();

		void clear();

		void add_point_mass(geometry* geo, pointmass* pmass);

		bool initialize();

		void calculateMassMatrix(double mul = 1.0);
		void calculateForceVector();
		vector4<double> calculateInertiaForce(vector4<double>& ev, matrix3x3<double>& J, vector4<double>& ep);
		void oneStepAnalysis();

		void cpu_run();
		void gpu_run();

		virtual bool initialize_simulation();
		virtual bool RunSim();

	private:
		unsigned nmass;

		matrix<double> *lhs;
		algebra::vector<double> *rhs;
		sparse_matrix<double> *cjaco;

		mbd_force* mforce;

		std::map<std::string, pointmass*>::iterator pm;
	};	
}

#endif