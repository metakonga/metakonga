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
			for (int i = 0; i < 4; i++){
				if (mple_e(i))
					use_p[3 + i] = true;
				else
					use_p[3 + i] = false;
			}
			maxnnz += 4;
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

double drivingConstraint::vel_rev_CE(double time)
{
	double pi = 0 + 0.2 * time;
	double e0 = cos(0.5 * pi);
	double v = kconst->iMass()->getEP().e0 - e0;
// 	mass* m = kconst->iMass();
// 	EPD v1 = m->getEP() - (init_e + time * mple_e);
	return v;//v1.length();
}

double drivingConstraint::constraintEquation(double ct)
{
	double ce = 0.0;
	ce = (this->*ce_func)(ct);

	return ce;
}