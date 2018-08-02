// #ifndef DEMBDSIMULATION_H
// #define DEMBDSIMULATION_H
// 
// #include "msimulation.h"
// #include "dem_simulation.h"
// #include "mbd_simulation.h"
// 
// class dembd_simulation : public simulation
// {
// public:
// 	dembd_simulation();
// 	dembd_simulation(modeler* _md, dem_simulation* _dem, mbd_simulation* _mbd);
// 	virtual ~dembd_simulation();
// 
// 	virtual bool initialize(bool isCpu);
// 
// 	bool saveResult(double t, unsigned int part);
// 
// public slots:
// 	virtual bool cpuRun();
// 	virtual bool gpuRun();
// 
// private:
// 	void predictionStep(double dt);
// 	void collision_dembd(double dt);
// 	void correctionStep(double dt);
// 	dem_simulation *dem;
// 	mbd_simulation *mbd;
// };
// 
// #endif