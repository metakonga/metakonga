#include "Integrator.h"

using namespace parSIM;

Integrator::Integrator(Simulation *_sim)
	: dt(0)
	, np(0)
	, pos(NULL)
	, vel(NULL)
	, acc(NULL)
	, omega(NULL)
	, alpha(NULL)
	, force(NULL)
	, moment(NULL)
	, sim(_sim)
{
	//dt = sim->getDt();
}

Integrator::~Integrator()
{

}

void Integrator::binding_data(vector4<double> *p, vector4<double> *v, vector4<double> *a, vector4<double> *omg, vector4<double> *aph, vector3<double> *f, vector3<double> *m)
{
	pos = p;
	vel = v;
	acc = a;
	omega = omg;
	alpha = aph;
	force = f;
	moment = m;
}

void Integrator::cu_binding_data(double* p, double *v, double* a, double* omg, double* aph, double* f, double* m)
{
	d_pos = p;
	d_vel = v;
	d_acc = a;
	d_omega = omg;
	d_alpha = aph;
	d_force = f;
	d_moment = m;
}

