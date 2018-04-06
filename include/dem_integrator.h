#ifndef DEM_INTEGRATOR_H
#define DEM_INTEGRATOR_H

class modeler;

class dem_integrator
{
public:
	dem_integrator();
	dem_integrator(modeler* _md);
	virtual ~dem_integrator();

	//virtual void updatePosition(double dt) = 0;
	//virtual void updateVelocity(double dt) = 0;
	virtual void updatePosition(double* dpos, double* dvel, double* dacc, unsigned int np) = 0;
	virtual void updateVelocity(
		  double *dvel, double* dacc
		, double *domega, double* dalpha
		, double *dforce, double* dmoment
		, double *dmass, double* dinertia, unsigned int np) = 0;

protected:
	double dt;
	modeler *md;
};

#endif