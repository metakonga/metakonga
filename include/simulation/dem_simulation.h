#ifndef DEM_SIMULATION_H
#define DEM_SIMULATION_H

#include "msimulation.h"
#include "neighborhood_cell.h"
#include "velocity_verlet.h"

class dem_simulation : public simulation
{
public:
	dem_simulation();
	dem_simulation(modeler *_md);
	virtual ~dem_simulation();

	virtual bool initialize(bool isCpu);
	bool saveResult(float ct, unsigned int p);
	bool cuSaveResult(float ct, unsigned int p);
	velocity_verlet* getIterator() { return itor; }
	neighborhood_cell* getNeighborhood() { return gb; }

private:
	
	void collision_dem(float dt);
	void cuCollision_dem();

	neighborhood_cell *gb;
	velocity_verlet *itor;

public slots:
	virtual bool cpuRun();
	virtual bool gpuRun();
};

#endif