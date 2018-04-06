#include "drivingConstraint.h"
#include "kinematicConstraint.h"
#include <QStringList>

drivingConstraint::drivingConstraint()
	: maxnnz(0)
{

}

drivingConstraint::drivingConstraint(QString& _name)
	: name(_name)
	, maxnnz(0)
{
	memset(use_p, 0, sizeof(bool) * 7);
}

drivingConstraint::~drivingConstraint()
{

}

void drivingConstraint::define(kinematicConstraint* kc, tDriving td, double val)
{
	direction = kc->iMass()->toGlobal(kc->h_i());
	direction.y = 0;
	EPD ep = kc->iMass()->getEP();
	mple_e = ep.w2ev(val * direction);
	scol = (kc->iMass()->ID() - 1) * 7;
	kconst = kc;
	dtype = td;
	if (dtype == DRIVING_DISPLACEMENT){

	}
	else if (dtype == DRIVING_VELOCITY){
		if (kconst->constType() == REVOLUTE) {
			update_func = &drivingConstraint::updateEV;
			ce_func = &drivingConstraint::vel_rev_CE;
			init_e = kc->iMass()->getEP();
			use_p[3] = true;
			maxnnz += 4;
		}
		if (kconst->constType() == TRANSLATIONAL) {
			init_t = kc->iMass()->getPosition();
			update_func = &drivingConstraint::updateV;
			ce_func = &drivingConstraint::vel_tra_CE;
			maxnnz += 3;
			if (direction.x) use_p[0] = true;
			if (direction.y) use_p[1] = true;
			if (direction.z) use_p[2] = true;
			//kc->iMass()->setVelocity(0.2 * direction);
		}
	}
}

void drivingConstraint::driving(double time)
{
	(this->*update_func)(time);
}

void drivingConstraint::updateEV(double time)
{
	double pi = 0 + 0.2 * time;
	double e0 = cos(0.5 * pi);

	EPD ep = kconst->iMass()->getEP();
	ep.e0 = e0;
	ep.e3 = sin(0.5 * pi);
	kconst->iMass()->setEP(ep);
// 	EPD new_ep = init_e + time * mple_e;
// 	kconst->iMass()->setEV(mple_e);
// 	kconst->iMass()->setEP(new_ep);
}

void drivingConstraint::updateV(double time)
{
	double p = 0 + 0.2 * time;
	VEC3D pos = p * direction;
	kconst->iMass()->setPosition(init_t + pos);
}

double drivingConstraint::vel_rev_CE(double time)
{
	double pi = 0 + 0.2 * time;
	double e0 = cos(0.5 * pi);
	double v = kconst->iMass()->getEP().e0 - e0;
// 	mass* m = kconst->iMass()rnr
// 	EPD v1 = m->getEP() - (init_e + time * mple_e);
	return v;//v1.length();
}

double drivingConstraint::vel_tra_CE(double time)
{
	VEC3D ne = VEC3D(init_t.x, 0, 0) + 0.2 * time * direction;
	VEC3D ipos = kconst->iMass()->getPosition();
	double out = ipos.x - ne.x;
	return out;
}

double drivingConstraint::constraintEquation(double ct)
{
	double ce = 0.0;
	ce = (this->*ce_func)(ct);

	return ce;
}