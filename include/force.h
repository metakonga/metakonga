#ifndef FORCE_H
#define FORCE_H

namespace parSIM
{
	class Simulation;
	class force
	{
	public:
		force(Simulation* Sim);
		force(Simulation* demSim, Simulation* mbdSim);
		~force();

		void setGravity(double x, double y, double z) { gravity = algebra::vector3<double>(x,y,z); }
		algebra::vector3<double>& Gravity() { return gravity; }

		vector3<double>* getForce() { return m_force; }
		vector3<double>* getMoment() { return m_moment; }
		double* cu_Force() { return d_force; }
		double* cu_Moment() { return d_moment; }

		contact_coefficient getCoefficient(std::string n){ return (coefficients.find(n)->second); }

		virtual void initialize();
		virtual void collision(cell_grid *detector, vector4<double> *pos, vector4<double> *vel, vector4<double> *acc, vector4<double> *omega = 0, vector4<double> *alpha = 0);
		virtual void collision(cell_grid *detector);
		//virtual void collision(cell_grid *detector, particle* pars) = 0;
		virtual void cu_collision(cell_grid *detector, double* pos, double* vel, double* acc, double* omega = 0, double* alpha=0, unsigned int cRun = 0);

		static double cohesive;
	protected:
		unsigned int m_np;
		double m_dt;

		Simulation *sim;
		Simulation *dem_sim;
		Simulation *mbd_sim;

		std::map<std::string, contact_coefficient> coefficients;
		std::map<int, geometry*> shapes;

		vector3<double> *m_force;
		vector3<double> *m_moment;

		double *d_force;
		double *d_moment;

		algebra::vector3<double> gravity;
	};
}

#endif