#ifndef FORCEELEMENT_H
#define FORCEELEMENT_H

#include "mbd_model.h"

class forceElement
{
public:
	enum Type{ TSDA = 0, RSDA, AXIAL_FORCE, AXIAL_ROTATION };

	forceElement();
	forceElement(
		QString _name, mbd_model* _md, Type tp, 
		pointMass* _b, pointMass* _a);
	virtual ~forceElement();

	QString Name();
	pointMass* BaseBody();
	pointMass* ActionBody();
	Type ForceType();
	virtual void calcForce(VECD* rhs) = 0;
	virtual void derivate(MATD& lhs, double mul) = 0;
	virtual void derivate_velocity(MATD& lhs, double mul) = 0;
	virtual void saveData(QTextStream& qts) = 0;

protected:
	QString name;

	pointMass* base;
	pointMass* action;
	Type type;

	mbd_model* md;
};

#endif