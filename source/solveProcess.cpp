#include "solveProcess.h"
#include "dem_simulation.h"
#include "neighborhood_cell.h"
#include "velocity_verlet.h"

solveProcess::solveProcess(/*modeler *_md, float _et, float _dt, unsigned int _step, */QObject *parent)
	: QObject(parent)
// 	, dem(NULL)
// 	, md(_md)
// 	, et(_et)
// 	, dt(_dt)
// 	, step(_step)
{

}
// 
// solveProcess::~solveProcess()
// {
// 
// }

void solveProcess::mainLoop()
{
// 	QProgressBar *pbar = new QProgressBar;
// 	pbar->setMaximum(100);
// 	pbar->setValue(0);
// 	pbar->show();
// 	neighborhood_cell *neigh = new neighborhood_cell("detector", md);
// 	neigh->setWorldOrigin(VEC3F(-1.0f, -1.0f, -1.0f));
// 	neigh->setGridSize(VEC3UI(128, 128, 128));
// 	neigh->setCellSize(md->particleSystem()->maxRadius() * 2.0f);
// 
// 	velocity_verlet *vv = new velocity_verlet(md);
// 	dem = new dem_simulation(md, neigh, vv);
}

void solveProcess::abort()
{

}