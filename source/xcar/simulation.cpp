#include "simulation.h"

double simulation::init_dt = 0.0;
double simulation::dt = 0.0;
simulation::deviceType simulation::dev = CPU;
double simulation::ctime = 0.0;
double simulation::et = 0.0;
double simulation::start_time = 0.0;
unsigned int simulation::st = 0;
unsigned int simulation::nstep = 0;

simulation::simulation()
//	: init_dt(0)
{

}

simulation::~simulation()
{

}

bool simulation::isCpu()
{
	return dev == CPU;
}

bool simulation::isGpu()
{
	return dev == GPU;
}

void simulation::setCPUDevice()
{
	dev = CPU;
}

void simulation::setGPUDevice()
{
	dev = GPU;
}

void simulation::setTimeStep(double _dt)
{
	dt = _dt;
}

void simulation::setCurrentTime(double _ct)
{
	ctime = _ct;
}

void simulation::setStartTime(double _st)
{
	start_time = _st;
}
