#include "Simulation.h"

double Simulation::dt = 0.0;
double Simulation::times = 0.0;

Simulation::Simulation(std::string bpath, std::string cname)
	: work_directory(bpath)
	, case_name(cname)
{

}

Simulation::~Simulation()
{

}