#ifndef SPRINGDAMPERMODEL_H
#define SPRINGDAMPERMODEL_H

#include "forceElement.h"
#include <QFile>
#include <QTextStream>

class springDamperModel : public forceElement
{
public:
	springDamperModel();
	springDamperModel(
		QString _n, mbd_model* _md, 
		pointMass* _b, VEC3D _baseLoc,
		pointMass* _a, VEC3D _actionLoc,
		double _k, double _c);
	virtual ~springDamperModel();

	virtual void calcForce(VECD* rhs);
	virtual void derivate(MATD& lhs, double mul);
	virtual void derivate_velocity(MATD& lhs, double mul);
	virtual void saveData(QTextStream& qts);

	void saveResult(double t);

private:
	VEC3D baseLoc;
	VEC3D actionLoc;
	VEC3D spi;
	VEC3D spj;
	double init_l;
	double k;
	double c;

	VEC3D L;
	double f;
	double l;
	double dl;
};

#endif