#ifndef DEM_FORCE_H
#define DEM_FORCE_H

#include "types.h"
#include <string>
#include <map>
#include "force.h"
#include "cell_grid.h"

using namespace algebra;

namespace parSIM
{
	class Simulation;

	class dem_force : public force
	{
	public:
		dem_force(Simulation *Sim);
		dem_force(Simulation *demSim, Simulation *mbdSim);
		virtual ~dem_force();

		virtual void initialize();
		virtual void collision(cell_grid *detector, vector4<double>* pos, vector4<double>* vel, vector4<double>* acc, vector4<double>* omega, vector4<double>* alpha);
		virtual void cu_collision(cell_grid *detector, bool* isLineContact, double* pos, double* vel, double* acc, double* omega, double* alpha, unsigned int cRun = 0);

		bool calForce(double ri, double rj, vector3<double>& posi, vector3<double>& posj, vector3<double>& veli, vector3<double>& velj, vector3<double>& omegai, vector3<double>& omegaj, vector3<double>& force, vector3<double>& moment);

		
		
// 		vector3<double>* getForce() { return m_force; }
// 		vector3<double>* getMoment() { return m_moment; }
// 		double* cu_Force() { return d_force; }
// 		double* cu_Moment() { return d_moment; }
// 		void setGravity(double x, double y, double z) { gravity = algebra::vector3<double>(x,y,z); }
// 		algebra::vector3<double>& Gravity() { return gravity; }
// 		contact_coefficient getCoefficient(geometry* geo = NULL){ return (coefficients.find(geo)->second); }
// 
// 		std::map<geometry*, contact_coefficient> coefficients;
// 		std::map<int, geo::shape*> shapes;

	private:
// 		unsigned int m_np;
// 		double m_dt;
// 		
// 		Simulation *sim;

// 		vector3<double> *m_force;
// 		vector3<double> *m_moment;

// 		double *d_force;
// 		double *d_moment;
	};
}

#endif