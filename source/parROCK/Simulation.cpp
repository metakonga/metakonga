#include "Simulation.h"

double Simulation::dt = 0.0;
double Simulation::times = 0.0;

Simulation::Simulation(std::string bpath, std::string cname)
	: base_path(bpath)
	, case_name(cname)
	, sort(NULL)
	, balls(NULL)
	, specificData("")
{
	sort = new sorter(this);
	gravity = vector3<double>(0.0, -9.80665, 0.0);
}

Simulation::~Simulation()
{
	if (geometries.size())
		for (std::map<std::string, Geometry*>::iterator it = geometries.begin(); it != geometries.end(); it++)
			delete it->second;

	if (objects.size())
		for (std::map<std::string, Object*>::iterator it = objects.begin(); it != objects.end(); it++)
			delete it->second;

	if (sort) delete sort; sort = NULL;
	if (balls) delete[] balls; balls = NULL;
}

double Simulation::CalMaxRadius()
{
	double maxRadius = 0;
	for (ball *b = ball::BeginBall(); b != NULL; b = b->NextBall())
	{
		maxRadius = maxRadius > b->Radius() ? maxRadius : b->Radius();
	}
	return maxRadius;
}

//bool Simulation::InsertContactCondition(ball* ib, ball* jb, double dist, vector3<double> nor)
//{
//	//if (dist <= 0) return false;
//	//contact c;
//	//c.SetIBall(ib);
//	//c.SetJBall(jb);
//	//c.CalculateContactForces(dist, nor);
//	//ib->ContactList().push_back(c);
//	//return true;
//}