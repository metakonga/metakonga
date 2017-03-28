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
	bool cuSaveResult(double ct, unsigned int p);
	velocity_verlet* getIterator() { return itor; }
	neighborhood_cell* getNeighborhood() { return gb; }

private:
	void collision_dem(double dt);
	void cuCollision_dem();

	device_parameters* paras;
	neighborhood_cell *gb;
	velocity_verlet *itor;

public slots:
	virtual bool cpuRun();
	virtual bool gpuRun();
};

#endif