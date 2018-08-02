#ifndef DEM_INTEGRATOR_H
#define DEM_INTEGRATOR_H

//class modeler;

class dem_integrator
{
public:
	enum Type { VELOCITY_VERLET };
	dem_integrator();
	dem_integrator(Type t);
	virtual ~dem_integrator();

	Type integrationType() { return type; }

	virtual void updatePosition(double* dpos, double* dvel, double* dacc, unsigned int np) = 0;
	virtual void updateVelocity(
		  double *dvel, double* dacc
		, double *domega, double* dalpha
		, double *dforce, double* dmoment
		, double *dmass, double* dinertia, unsigned int np) = 0;

protected:
	Type type;
};

#endif