#include "forceCalculator.h"
#include "Simulation.h"
#include "geometry.h"

using namespace parSIM;
bool forceCalculator::OnGravityForce = true;

forceCalculator::forceCalculator()
	: isAppliedForce(false)
{

}

forceCalculator::forceCalculator(vector3d& gra)
	: isAppliedForce(false)
{
	gravity = gra;
}

forceCalculator::~forceCalculator()
{

}

void forceCalculator::calculateForceVector(algebra::vector<double>& rhs, std::map<std::string, pointmass*>& masses)
{
 	rhs.zeros();
 	int nmass = masses.size();
 	int i=1;
 	int cnt=0;
 	vector3d nf = vector3<double>(0.0);
 	vector4d rf = vector4<double>(0.0);
	std::map<std::string, pointmass*>::iterator it;
	for(it = masses.begin(); it != masses.end(); it++){
		pointmass* mass = it->second;
		if(!mass->ID()) continue;
		geo::shape *sh = dynamic_cast<geo::shape*>(mass->Geometry());
		mass->Force() = sh->body_force;
		nf = OnGravityForce ? mass->Mass() * Simulation::gravity : 0.0;
		rf = calculateInertiaForce(mass->dOrientation(), mass->Inertia(), mass->Orientation());
		rhs.insert(cnt, POINTER3(nf), POINTER4(rf), 3, 4);
		cnt += 7;
	}
	if(isAppliedForce){
		vector3<double> af = 0.0;
		std::vector<appliedForceElement>::iterator aforce;
		for(aforce = aforces->begin(); aforce != aforces->end(); aforce++){
			pointmass* mass = (*(&masses.begin() + aforce->targetBody))->second;
			af = mass->toGlobal(aforce->aForce(Simulation::time) * vector3<double>(aforce->direction[0], aforce->direction[1],aforce->direction[2]));
			rhs.plus((aforce->targetBody - 1) * 7, POINTER3(af), 3);
		}
	}
}

vector4<double> forceCalculator::calculateInertiaForce(euler_parameter<double>& ev, matrix3x3<double>& J, euler_parameter<double>& ep)
{
	double GvP0 = -ev.e1*ep.e0 + ev.e0*ep.e1 + ev.e3*ep.e2 - ev.e2*ep.e3;
	double GvP1 = -ev.e2*ep.e0 - ev.e3*ep.e1 + ev.e0*ep.e2 + ev.e1*ep.e3;
	double GvP2 = -ev.e3*ep.e0 + ev.e2*ep.e1 - ev.e1*ep.e2 + ev.e0*ep.e3;
	return vector4<double>(
		-ev.e1*J.a00*GvP0 - ev.e2*J.a11*GvP1 - ev.e3*J.a22*GvP2,
		 ev.e0*J.a00*GvP0 - ev.e3*J.a11*GvP1 + ev.e2*J.a22*GvP2,
		 ev.e3*J.a00*GvP0 + ev.e0*J.a11*GvP1 - ev.e1*J.a22*GvP2,
		-ev.e2*J.a00*GvP0 + ev.e1*J.a11*GvP1 + ev.e0*J.a22*GvP2);
}