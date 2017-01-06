#include "DEM/DemSimulation.h"
#include "vparticles.h"
#include "glwidget.h"
#include "timer.h"
#include "vcontroller.h"
#include <cmath>
#include <QLineEdit>

//#include <QtWidgets>

DemSimulation::DemSimulation(parview::GLWidget *_gl)
	: gl(_gl), viewPars(NULL)
	// 		, pos(NULL), vel(NULL), acc(NULL), omega(NULL)
	// 		, alpha(NULL), force(NULL), moment(NULL)
	, sorted_id(NULL), cell_id(NULL), body_id(NULL)
	, cell_start(NULL), cell_end(NULL)
	, nParticle(0), nShapeNode(0), nGrid(0)
{
	//viewPars = dynamic_cast<parview::particles*>(gl->getViewParticle());
	//Initialize();
}

DemSimulation::~DemSimulation()
{
	// 		if (pos) delete[] pos; pos = NULL;
	// 		if (vel) delete[] vel; vel = NULL;
	// 		if (acc) delete[] acc; acc = NULL;
	// 		if (omega) delete[] omega; omega = NULL;
	// 		if (alpha) delete[] alpha; alpha = NULL;
	// 		if (force) delete[] force; force = NULL;
	// 		if (moment) delete[] moment; moment = NULL;
	if (cell_id) delete[] cell_id; cell_id = NULL;
	if (body_id) delete[] body_id; body_id = NULL;
	if (sorted_id) delete[] sorted_id; sorted_id = NULL;
	if (cell_start) delete[] cell_start; cell_start = NULL;
	if (cell_end) delete[] cell_end; cell_end = NULL;
}

bool DemSimulation::Initialize()
{
	// 		pos = new vector4<float>[viewPars->Np()];
	// 		vel = new vector4<float>[viewPars->Np()];
	// 		acc = new vector4<float>[viewPars->Np()];
	// 		omega = new vector4<float>[viewPars->Np()];
	// 		alpha = new vector4<float>[viewPars->Np()];
	// 		force = new vector3<float>[viewPars->Np()];
	// 		moment = new vector3<float>[viewPars->Np()];
	// 
	// 		for (unsigned int i = 0; i < viewPars->Np(); i++){
	// 			pos[i] = viewPars->getPositionToV4<float>(i);
	// 			vel[i] = viewPars->getVelocityToV4<float>(i);
	// 			acc[i].w = viewPars->Material().density * 4.0f * (float)M_PI * pow(pos[i].w, 3) / 3.0f;
	// 			alpha[i].w = 2.0f * acc[i].w * pow(pos[i].w, 2) / 5.0f;
	// 		}
// 	BaseSimulation::pBar = new QProgressBar;
// 	//pBar->setGeometry()
// 	BaseSimulation::durationTime = new QLineEdit;
// 	durationTime->setFrame(false);
// 	cellSize = viewPars->GetMaxRadius() * 2.0f;
// 	worldOrigin = vector3<float>(-1.1, -1.1, -1.1);
// 	gridSize = vector3<unsigned int>(128, 128, 128);
// 	nGrid = gridSize.x * gridSize.y * gridSize.z;
// 	parview::view_controller::setTotalFrame(0);
// 	cell_id = new unsigned int[viewPars->Np()]; memset(cell_id, 0, sizeof(unsigned int)*viewPars->Np());
// 	body_id = new unsigned int[viewPars->Np()]; memset(body_id, 0, sizeof(unsigned int)*viewPars->Np());
// 	sorted_id = new unsigned int[viewPars->Np()]; memset(sorted_id, 0, sizeof(unsigned int)*viewPars->Np());
// 	cell_start = new unsigned int[nGrid]; memset(cell_start, 0, sizeof(unsigned int)*nGrid);
// 	cell_end = new unsigned int[nGrid]; memset(cell_end, 0, sizeof(unsigned int)*nGrid);
	return true;
}

unsigned int DemSimulation::calcGridHash(algebra::vector3<int>& cell3d)
{
	algebra::vector3<int> gridPos;
	gridPos.x = cell3d.x & (gridSize.x - 1);
	gridPos.y = cell3d.y & (gridSize.y - 1);
	gridPos.z = cell3d.z & (gridSize.z - 1);
	return (gridPos.z*gridSize.y) * gridSize.x + (gridPos.y*gridSize.x) + gridPos.x;
}

void DemSimulation::reorderDataAndFindCellStart(unsigned ID, unsigned begin, unsigned end)
{
	cell_start[ID] = begin;
	cell_end[ID] = end;
	unsigned dim = 0, bid = 0;
	for (unsigned i(begin); i < end; i++){
		sorted_id[i] = body_id[i];
	}
}

// void DemSimulation::TimeStep(float dt, bool seq)
// {
// 	// 		pos = new vector4<float>[viewPars->Np()];
// 	// 		vel = new vector4<float>[viewPars->Np()];
// 	// 		acc = new vector4<float>[viewPars->Np()];
// 	// 		omega = new vector4<float>[viewPars->Np()];
// 	// 		alpha = new vector4<float>[viewPars->Np()];
// 	// 		force = new vector3<float>[viewPars->Np()];
// 	// 		moment = new vector3<float>[viewPars->Np()];
// 	// 
// 	// 		for (unsigned int i = 0; i < viewPars->Np(); i++){
// 	// 			pos[i] = viewPars->getPositionToV4<float>(i);
// 	// 			vel[i] = viewPars->getVelocityToV4<float>(i);
// 	// 			acc[i].w = viewPars->Material().density * 4.0f * (float)M_PI * pow(pos[i].w, 3) / 3.0f;
// 	// 			alpha[i].w = 2.0f * acc[i].w * pow(pos[i].w, 2) / 5.0f;
// 	// 		}
// }

void DemSimulation::CpuRun()
{
// 	algebra::vector4<float> *pos = new algebra::vector4<float>[viewPars->Np()];
// 	algebra::vector4<float> *vel = new algebra::vector4<float>[viewPars->Np()];
// 	algebra::vector4<float> *acc = new algebra::vector4<float>[viewPars->Np()];
// 	algebra::vector4<float> *omega = new algebra::vector4<float>[viewPars->Np()];
// 	algebra::vector4<float> *alpha = new algebra::vector4<float>[viewPars->Np()];
// 	algebra::vector3<float> *force = new algebra::vector3<float>[viewPars->Np()];
// 	algebra::vector3<float> *moment = new algebra::vector3<float>[viewPars->Np()];
// 
// 	for (unsigned int i = 0; i < viewPars->Np(); i++){
// 		pos[i] = viewPars->getPositionToV4<float>(i);
// 		//vel[i] = viewPars->getVelocityToV4<float>(i);
// 		acc[i].w = viewPars->Material().density * 4.0f * (float)M_PI * pow(pos[i].w, 3) / 3.0f;
// 		alpha[i].w = 2.0f * acc[i].w * pow(pos[i].w, 2) / 5.0f;
// 		force[i] = acc[i].w*gravity;
// 	}
// 	itor = VELOCITY_VERLET;
// 	unsigned int part = 0;
// 	unsigned int cStep = 0;
// 	unsigned int eachStep = 0;
// 	unsigned int nStep = static_cast<unsigned int>((simTime / dt) + 1);
// 	pBar->setMaximum(nStep / saveStep);
// 	pBar->setValue(part);
// 	parSIM::timer tmer;
// 	time_t t;
// 	tm date;
// 	std::time(&t);
// 	localtime_s(&date, &t);
// 	float times = cStep * dt;
// 	parview::view_controller::addTimes(parview::view_controller::getTotalBuffers(), times);
// 	float elapsed_time = 0.f;
// 	cStep++;
// 	tmer.Start();
// 	QString durt_str;
// 	float durt = 0;
// 	viewPars->insert_particle_buffer(&(pos[0].x), &(vel[0].x), &(force[0].x), &(moment[0].x), viewPars->Np(), part);
// 	while (nStep > cStep){
// 		times = cStep * dt;
// 		TimeStepping<float, true>(pos, vel, acc, omega, alpha, force, moment);
// 		
// 		ContactDetect(pos);
// 		Collision(pos, vel, acc, omega, alpha, force, moment);
// 		TimeStepping<float, false>(pos, vel, acc, omega, alpha, force, moment);
// 		if (!((cStep) % saveStep)){
// 			part++;
// 			//parview::view_controller::addTimes(part++, times);
// 			pBar->setValue(part);
// 			time(&t);
// 			localtime_s(&date, &t);
// 			tmer.Stop();
// 			durt += tmer.GetElapsedTimeF();
// 			parview::view_controller::upBufferCount();
// 			parview::view_controller::addTimes(parview::view_controller::getTotalBuffers(), times);
// 			viewPars->insert_particle_buffer(&(pos[0].x), &(vel[0].x), &(force[0].x), &(moment[0].x), viewPars->Np(), part);
// 			durt_str.sprintf("%.4f", durt);
// 			durationTime->setText(durt_str);
// 			eachStep = 0;
// 			tmer.Start();
// 		}
// 		cStep++;
// 		eachStep++;
// 	}
// 	delete[] pos;
// 	delete[] vel;
// 	delete[] acc;
// 	delete[] omega;
// 	delete[] alpha;
// 	delete[] force;
// 	delete[] moment;
}

void DemSimulation::GpuRun()
{

}