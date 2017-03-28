#ifndef MASS_H
#define MASS_H

#include "vobject.h"
#include "vpolygon.h"
#include "mphysics_numeric.h"
#include <QTextStream>

class modeler;

class mass
{
public:

	mass();
	mass(modeler *_md, QString& _name);
	~mass();

	int ID() { return id; }
	QString& name(){ return nm; }
	double getMass() { return ms; }
	MAT33D getInertia() { return inertia; }
	VEC3D getPosition() { return pos; }
	EPD getEP() { return ep; }
	EPD getEV() { return ev; }
	EPD getEA() { return ea; }
	VEC3D getExternalForce() { return ef; }
	VEC3D getCollisionForce() { return cf; }
	VEC3D getExternalMoment() { return em; }
	VEC3D getCollisionMoment() { return cm; }
	VEC3D getVelocity() { return vel; }
	VEC3D getAcceleration() { return acc; }
	VEC3D getSymInertia() { return sym_iner; }
	VEC3D getPriInertia() { return prin_iner; }

	void setID(int _id) { id = _id; }
	//void setName(QString _n)
	void setMass(double _ms) { ms = _ms; }
	void setMassPoint(VEC3D _mp) { pos = _mp; }
	void setSymIner(VEC3D _si) { sym_iner = _si; }
	void setPrinIner(VEC3D _pi) { prin_iner = _pi; }
	void setInertia() { inertia.diagonal(POINTER3(prin_iner)); }
	void setPosition(VEC3D& _p) { pos = _p; }
	void setVelocity(VEC3D& _v) { vel = _v; }
	void setAcceleration(VEC3D& a) { acc = a; }
	void setExternalForce(VEC3D& _f) { ef = _f; }
	void setExternalMoment(VEC3D& _m) { em = _m; }
	void addExternalForce(VEC3D& _f) { ef += _f; }
	void addExternalMoment(VEC3D& _m) { em += _m; }
	void setCollisionForce(VEC3D& _f) { cf = _f; }
	void setCollisionMoment(VEC3D& _m) { cm = _m; }
	void addCollisionForce(VEC3D& _f) { cf += _f; }
	void addCollisionMoment(VEC3D& _m) { cm += _m; }
	void setEP(EPD& _ep) { ep = _ep; }
	void setEV(EPD& _ev) { ev = _ev; }
	void setEA(EPD& _ea) { ea = _ea; }
	void saveData(QTextStream& ts) const;
	void openData(QTextStream& ts);
	void makeTransformationMatrix();

	VEC3D toLocal(VEC3D &v);
	VEC3D toGlobal(VEC3D &v);

	void setBaseGeometryType(tObject _tobj) { baseGeometryType = _tobj; }
	void setGeometryObject(vobject* _vobj) { vobj = _vobj; }
	void setPolyGeometryObject(vpolygon* _vpobj) { vpobj = _vpobj; }
	tObject getBaseGeometryType() { return baseGeometryType; }
	vobject* getGeometryObject() { return vobj; }
	vpolygon* getPolyGeometryObject() { return vpobj; }

private:
	tObject baseGeometryType;
	QString nm;
	int id;
	double ms;				// mass of object
	VEC3D sym_iner;			// Ixx, Iyy, Izz
	VEC3D prin_iner;		// Ixy, Ixz, Iyz
	MAT33D inertia;

	VEC3D pos;
	VEC3D vel;
	VEC3D acc;
	VEC3D ef;
	VEC3D em;
	VEC3D cf;
	VEC3D cm;

	EPD ep;					// euler parameter
	EPD ev;					// angular velocity of ep
	EPD ea;					// angular acceleration of ep

	MAT33D A;				// transformation matrix	

	modeler *md;			// pointer of modeler
	vobject* vobj;
	vpolygon* vpobj;
	//object* obj;
};

#endif