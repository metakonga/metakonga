#include "drivingConstraint.h"
#include "kinematicConstraint.h"
#include "simulation.h"
#include "numeric_utility.h"
#include <QStringList>
#include <QDebug>
drivingConstraint::drivingConstraint()
	: maxnnz(0)
	, theta(0)
	, start_time(0.0)
	, plus_time(0.0)
{

}

drivingConstraint::drivingConstraint(QString _name)
	: name(_name)
	, maxnnz(0)
	, theta(0)
	, start_time(0.0)
	, plus_time(0.0)
{

}

drivingConstraint::~drivingConstraint()
{

}

void drivingConstraint::define(
	kinematicConstraint* kc, drivingConstraint::Type td, double init, double cont)
{
//	m = kc->jMass();
//	pos0 = m->Position();
	kconst = kc;
// 	qDebug() << kconst->name();
// 	qDebug() << (int)td;
// 	qDebug() << init;
// 	qDebug() << cont;
	type = td;
	init_v = init;
	cons_v = cont;
	if (td == DRIVING_ROTATION)
	{
		pointMass* im = kconst->iMass();
		pointMass* jm = kconst->jMass();
		VEC3D fi = im->toGlobal(kconst->f_i());
		VEC3D fj = jm->toGlobal(kconst->f_j());
		init_v = acos(fi.dot(fj));;
	}
// 	else
// 	{
// 		VEC3D dist = kconst->CurrentDistance();
// 		init_v = dist.length();
// 	}
//	cons_v = cont;
}

void drivingConstraint::updateInitialCondition()
{
	if (type == DRIVING_ROTATION)
	{
		pointMass* im = kconst->iMass();
		pointMass* jm = kconst->jMass();
		VEC3D fi = im->toGlobal(kconst->f_i());
		VEC3D fj = jm->toGlobal(kconst->f_j());
		init_v = acos(fi.dot(fj));
	}
	else
	{
		VEC3D dist = kconst->CurrentDistance();
		init_v = dist.length();
	}
}

void drivingConstraint::constraintEquation(double mul, double* rhs)
{
	double v = 0.0;
	if (type == DRIVING_TRANSLATION)
	{
		VEC3D dist = kconst->CurrentDistance();
		VEC3D hi = kconst->iMass()->toGlobal(kconst->h_i());
		if (start_time > simulation::ctime + plus_time)
			v = hi.dot(dist) - (init_v + 0.0 * simulation::ctime);
		else
			v = hi.dot(dist) - (init_v + cons_v * (simulation::ctime - start_time + plus_time));
		rhs[srow] = mul * v;
	}
	else if (type == DRIVING_ROTATION)
	{
		if (start_time > simulation::ctime + plus_time)
			v = theta - (init_v + 0.0 * simulation::ctime);
		else
			v = theta - (init_v + cons_v * (simulation::ctime - start_time + plus_time));
		rhs[srow] = mul * v;
	}
}

void drivingConstraint::constraintJacobian(SMATD& cjaco)
{
	if (type == DRIVING_TRANSLATION)
	{
		VEC3D fdij = kconst->CurrentDistance();
		pointMass* im = kconst->iMass();
		pointMass* jm = kconst->jMass();
		VEC3D hi = im->toGlobal(kconst->h_i());
		VEC3D D1;
		VEC4D D2;
		int ic = kconst->iColumn();
		int jc = kconst->jColumn();
		if (im->MassType() != pointMass::GROUND)
		{
			D1 = -hi;
			D2 = transpose(fdij, B(im->getEP(), kconst->h_i())) - transpose(hi, B(im->getEP(), kconst->sp_i()));
			cjaco.extraction(srow, ic, POINTER(D1), POINTER(D2), VEC3_4);
		}
		if (jm->MassType() != pointMass::GROUND)
		{
			D1 = hi;
			transpose(hi, B(jm->getEP(), kconst->sp_j()));
			cjaco.extraction(srow, jc, POINTER(D1), POINTER(D2), VEC3_4);
		}
	}
	else if (type == DRIVING_ROTATION)
	{
		pointMass* im = kconst->iMass();
		pointMass* jm = kconst->jMass();
		int ic = kconst->iColumn();
		int jc = kconst->jColumn();
		VEC3D fi = im->toGlobal(kconst->f_i());
		VEC3D fj = jm->toGlobal(kconst->f_j());
		VEC3D gi = im->toGlobal(kconst->g_i());
		double stheta = acos(fi.dot(fj));
		double prad = 0.0;
// 		qDebug() << "start_time : " << start_time;
// 		qDebug() << "simulation_time : " << simulation::ctime;
		if (start_time > simulation::ctime + plus_time)
			prad = init_v + 0.0 * simulation::ctime;
		else
			prad = init_v + cons_v * (simulation::ctime - start_time + plus_time);
		if (prad > M_PI)
			stheta = stheta;
	//	qDebug() << "prad : " << prad;
		theta = numeric::utility::angle_coefficient(prad, stheta);
		VEC3D zv;
		VEC4D D1 = transpose(fj, cos(theta) * B(im->getEP(), kconst->g_i()));
		VEC4D D2 = transpose((cos(theta) * gi - sin(theta) * fi), B(jm->getEP(), kconst->f_j()));
		if (im->MassType() != pointMass::GROUND)
			cjaco.extraction(srow, ic, POINTER(zv), POINTER(D1), VEC3_4);
		if (jm->MassType() != pointMass::GROUND)
			cjaco.extraction(srow, jc, POINTER(zv), POINTER(D2), VEC3_4);
	}
	
}

void drivingConstraint::saveData(QTextStream& qts)
{
	qts << "ELEMENT " << "drive_constraint" << endl
		<< "NAME " << name << endl
		<< "TYPE " << type << endl
		<< "TARGET " << kconst->name() << endl
		<< "START_TIME " << start_time << endl
		<< "PARAMETES " << init_v << " " << cons_v << endl;
}
