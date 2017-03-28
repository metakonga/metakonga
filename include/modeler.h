#ifndef MODELER_H
#define MODELER_H

#include "particle_system.h"
#include "mphysics_types.h"
#include <QMap>
#include <QFile>
#include <QTextStream>
#include <QString>
//#include "object.h"

class cube;
class mass;
class plane;
class polygonObject;
class cylinder;
class kinematicConstraint;
class drivingConstraint;
class collision;
class collision_particles_particles;
class collision_particles_plane;
class GLWidget;
class database;

// QT_BEGIN_NAMESPACE
// class QPlainTextEdit;
// QT_END_NAMESPACE

class modeler
{
public:
	modeler();
	modeler(QString _name, tSimulation _sim, tUnit u, tGravity dg);
	~modeler();

	//void comm(QString com);
	//void bindingCommand(QPlainTextEdit* _pte) { pte = _pte; }
	void setParticleSystem(particle_system* _ps) { ps = _ps; }
	void setDatabase(database* _db) { db = _db; }
	mass* makeMass(QString _name);
	cube* makeCube(QString _name, tMaterial _mat, tRoll _roll);
	plane* makePlane(QString _name, tMaterial _mat, tRoll _roll);
	cylinder* makeCylinder(QString _name, tMaterial _mat, tRoll _roll);
	kinematicConstraint* makeKinematicConstraint(QString _name, tKinematicConstraint kt, mass* i, VEC3D& spi, VEC3D& fi, VEC3D& gi, mass* j, VEC3D& spj, VEC3D& fj, VEC3D& gj);
	drivingConstraint* makeDrivingConstraint(QString _name, kinematicConstraint* kconst, tDriving td, double val);
	polygonObject* makePolygonObject(tImport tm, QString file);
	particle_system* makeParticleSystem(QString _name);
	collision* makeCollision(QString _name, double _rest, double _fric, double _rfric, double _coh, double _ratio, tCollisionPair tcp, tContactModel tcm, void *o1, void *o2 = NULL);
	QMap<QString, object*>& objects() { return objs; }
	QMap<QString, collision*>& collisions() { return cs; }
	object* objectFromStr(QString& str) { return objs[str]; }
	unsigned int numPolygonSphere();
	unsigned int numParticle() { return ps ? ps->numParticle() : 0; }
	unsigned int numCollision() { return cs.size(); }
	unsigned int numMass() { return masses.size(); }
	QString& modelPath() { return model_path; }
	QString& modelName() { return name; }
	void actionDelete(const QString& tg);
	particle_system* particleSystem() { return ps; }
	QList<polygonObject*>& polyObjects() { return polygons; }
	QMap<QString, collision*>& collision_map() { return cs; }
	QMap<QString, kinematicConstraint*>& kinConstraint() { return consts; }
	QMap<QString, drivingConstraint*>& drivingConstraints() { return dconsts; }
	QMap<object*, mass*>& pointMasses() { return masses; }
	
	VEC3D gravity() { return grav; }

	void saveModeler();
	void openModeler(GLWidget *gl, QString& file);
	//void updateObject(float dt, tSolveDevice tsd = CPU);
	void runExpression(double ct, double dt);
	//void runDriving(double ct);
	template<typename object_type>
	object_type getChildObject(const QString& str)
	{
		return dynamic_cast<object_type>(objs.find(str).value());
	}

	unsigned int numCube() { return ncube; }
	unsigned int numPlane() { return nplane; }
	unsigned int numCylinder() { return ncylinder; }
	unsigned int numPoly() { return npoly; }

private:
	VEC3D grav;
	tUnit unit;								// unit
	tGravity dg;	// direction of gravity
	unsigned int ncube;
	unsigned int nplane;
	unsigned int ncylinder;
	unsigned int npoly;
	QString model_path;
	QString name;
	tSimulation tsim;

	QList<polygonObject*> polygons;
	QMap<QString, object*> objs;
	QMap<object*, mass*> masses;
	QMap<QString, collision*> cs;
	QMap<QString, kinematicConstraint*> consts;
	QMap<QString, drivingConstraint*> dconsts;
	//QPlainTextEdit *pte;
	particle_system *ps;
	database* db;
};

#endif