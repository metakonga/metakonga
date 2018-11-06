#include "axialRotationForce.h"
//#include "mbd_model.h"

axialRotationForce::axialRotationForce()
	: forceElement()
{

}

axialRotationForce::axialRotationForce(
	QString _n, mbd_model* _md, VEC3D _loc, VEC3D _u, pointMass* _a, pointMass* _b)
	: forceElement(_n, _md, AXIAL_ROTATION, _a, _b)
	, loc(_loc)
	, u(_u)
{
 	spi = _a->toGlobal(loc - base->Position());
 	spj = _b->toGlobal(loc - action->Position());
}

axialRotationForce::~axialRotationForce()
{

}

void axialRotationForce::setForceValue(double r_f)
{
	r_force = r_f;
}

void axialRotationForce::calcForce(VECD* rhs)
{
	VEC3D sp;
	if (base->MassType() != pointMass::GROUND)
	{
		sp = base->Position() - loc;
		if (sp.x)
			sp.x = 1.0 / sp.x;
		if (sp.y)
			sp.y = 1.0 / sp.y;
		if (sp.z)
			sp.z = 1.0 / sp.z;
		MAT33D m_sp(
			0.0, -sp.z, sp.y,
			sp.z, 0.0, -sp.x,
			-sp.y, sp.x, 0.0);
		VEC3D mf = m_sp * (r_force * u);
		int ic = base->ID() * base->NumDOF();
		(*rhs)(ic) = mf.x;
		(*rhs)(ic + 1) = mf.y;
		(*rhs)(ic + 2) = mf.z;
	}
	if (action->MassType() != pointMass::GROUND)
	{
		//sp;// = action->Position() - loc;
		if(spj.x)
			sp.x = 1.0 / spj.x;
		if(spj.y)
			sp.y = 1.0 / spj.y;
		if(spj.z)
			sp.z = 1.0 / spj.z;
		MAT33D m_sp(
			0.0, -sp.z, sp.y,
			sp.z, 0.0, -sp.x,
			-sp.y, sp.x, 0.0);
		MAT33D inv_sp = m_sp.inv();
		VEC3D mf = m_sp * (r_force * u);
	    mf = action->toGlobal(mf);
	///	double mag = mf.length();
		int ic = action->ID() * base->NumDOF();
		(*rhs)(ic) += mf.x;
		(*rhs)(ic + 1) += mf.y;
		(*rhs)(ic + 2) += mf.z;
	}
}

void axialRotationForce::saveData(QTextStream& qts)
{
	qts << endl
		<< "FORCE_ELEMENT " << type << endl
		<< "NAME " << name << endl
		<< "BASE " << base->Name() << endl
		<< "ACTION " << action->Name() << endl
		<< "LOCATION " << loc.x << " " << loc.y << " " << loc.z << endl
		<< "AXIS_UNIT " << u.x << " " << u.y << " " << u.z << endl
		<< "FORCE_VALUE " << r_force << endl;
}
