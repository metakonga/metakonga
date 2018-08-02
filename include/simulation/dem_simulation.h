#ifndef DEM_SIMULATION_H
#define DEM_SIMULATION_H

#include "simulation.h"
#include "grid_base.h"
#include "dem_model.h"
#include "contactManager.h"
#include "velocity_verlet.h"

class dem_simulation : public simulation
{
public:
	dem_simulation();
	dem_simulation(dem_model *_md);
	virtual ~dem_simulation();

	bool initialize(contactManager* _cm);
	bool oneStepAnalysis();
	QString saveResult(double *vp, double* vv, double ct, unsigned int pt);

private:
	void applyMassForce();
	void clearMemory();
	void allocationMemory(unsigned int _np);

private:
	unsigned int np;
	dem_model* md;
	grid_base* dtor;
	dem_integrator* itor;
	contactManager* cm;
	double *mass;
	double *inertia;
	double *pos;
	double *vel;
	double *acc;
	double *avel;
	double *aacc;
	double *force;
	double *moment;

	double *dmass;
	double *diner;
	double *dpos;
	double *dvel;
	double *dacc;
	double *davel;
	double *daacc;
	double *dforce;
	double *dmoment;
// 	virtual bool initialize(bool isCpu);
// 	bool saveResult(double ct, unsigned int p);
// 	bool savePartResult(double ct, unsigned int p);
// 	velocity_verlet* getIterator() { return itor; }
// 	neighborhood_cell* getNeighborhood() { return gb; }
// 	void cudaUpdatePosition();
// 	void cudaDetection();
// 	void cudaUpdateVelocity();
// 
// private:
// 	void clear();
// 
// 	void collision_dem(double dt);
// 	void cuCollision_dem();
// 	void cudaAllocMemory(unsigned int np);
// 
// 	unsigned int np;
// 
// 	device_parameters* paras;
// 	neighborhood_cell *gb;
// 	velocity_verlet *itor;
// 	dem_model* md;
// 
// 	VEC4D_PTR m_pos;
// 	VEC3D_PTR m_vel;
// 	VEC3D_PTR m_force;
// 
// 	double* d_pos;
// 	double* d_vel;
// 	double* d_acc;
// 	double* d_omega;
// 	double* d_alpha;
// 	double* d_fr;
// 	double* d_mm;
// 	double* d_ms;
// 	double* d_iner;
// 	double* d_riv;
// 	unsigned int* d_pair_riv;
};

#endif