#ifndef OBJECT_H
#define OBJECT_H

#include "types.h"
#include "algebraMath.h"
#include <QString>
#include <QTextStream>

class vobject;

class object
{
public:
	object();
	object(QString _name, geometry_type gt, geometry_use gu);
	object(const object& obj);
	virtual ~object();

	unsigned int ID() const { return id; }
	dimension_type NumDOF() { return dim; }
	QString Name() const { return name; }
	geometry_type ObjectType() const { return obj_type; }
	material_type MaterialType() const { return mat_type; }
	geometry_use RollType() const { return roll_type; }
	double Density() const { return d; }
	double Youngs() const { return y; }
	double Poisson() const { return p; }
	double Shear() const { return sm; }
	unsigned int particleCount() const { return count; }
	void setViewObject(vobject* vo) { vobj = vo; }
	void setViewMarker(vobject* vm) { marker = vm; }
	vobject* ViewObject() { return vobj; }
	vobject* ViewMarker() { return marker; }
	void updateView(VEC3D p, VEC3D r);
	
	double Volume() const { return vol; }
	void setVolume(double _vol) { vol = _vol; }

	void setRoll(geometry_use tr)  { roll_type = tr; }
	void setID(unsigned int _id) { id = _id; }
	void setMaterial(material_type _tm, double _y = 0, double _d = 0, double _p = 0, double _s = 0);

	VEC3D DiagonalInertia0() { return dia_iner0; }
	VEC3D SymetricInertia0() { return sym_iner0; }

	virtual void saveData(QTextStream& ts) {};

protected:
	static unsigned int count;
	dimension_type dim;
	unsigned int id;	
					// pointMass of object
 	VEC3D dia_iner0;			// Ixx, Iyy, Izz
 	VEC3D sym_iner0;		// Ixy, Ixz, Iyz
	double vol;
	QString name;
	geometry_use roll_type;
	geometry_type obj_type;
	material_type mat_type;
	double d;		// density
	double y;		// young's modulus
	double p;		// poisson ratio
	double sm;		// shear modulus

	vobject* vobj;
	vobject* marker;
};

#endif