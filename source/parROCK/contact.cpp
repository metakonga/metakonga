#include "contact.h"
#include "Simulation.h"
#include "WallElement.h"

double ccontact::prod = 0.0;

ccontact::ccontact(const ccontact& ct)
{
	iBall = ct.IBall();
	jBall = ct.JBall();
	cPoint = ct.ContactPoint();
	wall = ct.Wall();
	object = ct.Obj();
	nforce = ct.NormalForce();
	sforce = ct.ShearForce();
}

void ccontact::CalculateContactForces(double cdist, vector3<double>& nor, double kn, double ks, double fric)
{
	normal = nor;
	double kn_j = jBall ? jBall->Kn() : (wall ? wall->Kn() : kn);
	double ks_j = jBall ? jBall->Ks() : (wall ? wall->Ks() : ks);
	ekn = iBall->Kn() * kn_j / (iBall->Kn() + kn_j);
	eks = iBall->Ks() * ks_j / (iBall->Ks() + ks_j);
	vector3<double> xc = iBall->Position() + (iBall->Radius() - 0.5 * cdist) * nor;
	vector3<double> dx = xc - iBall->Position();
	vector3<double> Vi = (jBall ? jBall->Velocity() + jBall->Omega().cross(xc - jBall->Position()) : 0.0) - (iBall->Velocity() + iBall->Omega().cross(xc - iBall->Position()));
	vector3<double> Vi_s = Vi - Vi.dot(nor) * nor;
	vector3<double> dUi_s = Simulation::dt * Vi_s;
	vector3<double> dFi_s = -eks*dUi_s; //jBall ? -eks * dUi_s : eks * dUi_s;

	cPoint = xc;
	nforce = (ekn * pow(cdist, prod)) * nor;
	sforce += dFi_s;
}

ccontact* ccontact::c_b1clist()
{
	std::map<ball*, ccontact>::iterator it = iBall->ContactPMap().begin();
	return &(it->second);
}

ccontact* ccontact::c_b2clist()
{
	std::map<ball*, ccontact>::iterator it = iBall->ContactPMap().find(jBall);
	return &(it->second);
}