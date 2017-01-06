#include "driving_constraint.h"

using namespace parSIM;

drivingConstraint::drivingConstraint()
{

}

// drivingConstraint::drivingConstraint(int tb, char tc, double (*func)(double))
// {
// 	sRow = 0;
// 	for(int i(0); i < 3; i++) 
// 		translationIndex[i] = 0;
// 	targetBody = tb;
// 	targetCoord = tc;
// 	driving = func;
// 	initPos = make_vector3d(0.0);
// }

drivingConstraint::drivingConstraint(drivingType _dType, kinematicConstraint *_kconst, int tb, char tc, double (*func)(double), double (*d_func)())
{
	sRow = 0;
	for(int i(0); i < 3; i++) 
		translationIndex[i] = 0;
	dType = _dType;
	targetBody = tb;
	targetCoord = tc;
	driving = func;
	d_driving = d_func;
	kconst = _kconst;
	initPos = vector3<double>(0.0);
}

drivingConstraint::drivingConstraint(const drivingConstraint& driv)
{
	sRow = driv.getStartRow();
	for(int i(0); i < 3; i++) 
		translationIndex[i] = driv.translationIndex[i];
	dType = driv.dType;
	driving = driv.driving;
	d_driving = driv.d_driving;
	initPos = driv.initPos;
	kconst = driv.getTargetJoint();
	targetBody = driv.getTargetBody();
	targetCoord = driv.getTargetCoord();
}

drivingConstraint::~drivingConstraint()
{

}