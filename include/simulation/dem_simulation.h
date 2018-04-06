#ifndef DEM_SIMULATION_H
#define DEM_SIMULATION_H

#include "msimulation.h"
#include "neighborhood_cell.h"
#include "velocity_verlet.h"

struct device_parameters;

class dem_simulation : public simulation
{
public:
	dem_simulation();
	dem_simulation(modeler *_md);
	virtual ~dem_simulation();

	virtual bool initialize(bool isCpu);
	bool saveResult(double ct, unsigned int p);
	bool savePartResult(double ct, unsigned int p);
	velocity_verlet* getIterator() { return itor; }
	neighborhood_cell* getNeighborhood() { return gb; }
	void cudaUpdatePosition();
	void cudaDetection();
	void cudaUpdateVelocity();

private:
	void clear();

	void collision_dem(double dt);
	void cuCollision_dem();
	void cudaAllocMemory(unsigned int np);

	unsigned int np;

	device_parameters* paras;
	neighborhood_cell *gb;
	velocity_verlet *itor;

	VEC4D_PTR m_pos;
	VEC3D_PTR m_vel;
	VEC3D_PTR m_force;

	double* d_pos;
	double* d_vel;
	double* d_acc;
	double* d_omega;
	double* d_alpha;
	double* d_fr;
	double* d_mm;
	double* d_ms;
	double* d_iner;
	double* d_riv;
	unsigned int* d_pair_riv;

public slots:
	virtual bool cpuRun();
	virtual bool gpuRun();
};

#endif