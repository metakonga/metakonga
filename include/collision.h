#ifndef COLLISION_H
#define COLLISION_H

#include <QString>
#include "mphysics_types.h"
#include "modeler.h"
#include "grid_base.h"
//class object;
//class particle_system;
//class grid_base;

class collision
{
public:
	collision();
	collision(QString& _name, modeler* _md, QString& o1, QString& o2, tCollisionPair _tp = NO_COLLISION_PAIR);
	collision(const collision& cs);
	virtual ~collision();

	void setContactParameter(float _rest, float _sratio, float _fric, float _coh) { rest = _rest; sratio = _sratio; fric = _fric; coh = _coh; }
	void setGridBase(grid_base *_gb) { gb = _gb; }
	float cohesionForce(float ri, float rj, float Ei, float Ej, float pri, float prj, float Fn);

	virtual bool collid(float dt) = 0;
	virtual bool cuCollid() = 0;
	virtual bool collid_with_particle(unsigned int i, float dt) = 0;

	constant getConstant(float ir, float jr, float im, float jm, float iE, float jE, float ip, float jp,  float riv);
	grid_base* getGridBase() { return gb; }
	tCollisionPair getCollisionPairType() { return tcp; }

	void save_collision_data(QTextStream& ts);

	QString& firstObject() { return oname1; }
	QString& secondObject() { return oname2; }

	float cohesion() { return coh; }

protected:
	QString name;
	QString oname1;
	QString oname2;
	float rest;
	float sratio;
	float fric;		
	float coh;

	float *td;			// tangential displacement

	tContactModel tcm; 
	tCollisionPair tcp;

	modeler* md;
	grid_base* gb;
};

#endif