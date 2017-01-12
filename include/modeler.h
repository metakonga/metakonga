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
class collision;
class collision_particles_particles;
class collision_particles_plane;
class GLWidget;

class modeler
{
public:
	modeler();
	modeler(QString _name, tSimulation _sim, tUnit u, tGravity dg);
	~modeler();

	void setParticleSystem(particle_system* _ps) { ps = _ps; }
	mass* makeMass(QString _name);
	cube* makeCube(QString _name, tMaterial _mat, tRoll _roll);
	plane* makePlane(QString _name, tMaterial _mat, tRoll _roll);
	cylinder* makeCylinder(QString _name, tMaterial _mat, tRoll _roll);
	kinematicConstraint* makeKinematicConstraint(QString _name, tKinematicConstraint kt, VEC3D& loc, mass* i, VEC3D& fi, VEC3D& gi, mass* j, VEC3D& fj, VEC3D& gj);
	polygonObject* makePolygonObject(tImport tm, QString file);
	particle_system* makeParticleSystem(QString _name);
	collision* makeCollision(QString _name, float _rest, float _fric, float _rfric, float _coh, tCollisionPair tcp, void *o1, void *o2 = NULL);
	//collision_particles_plane* make_collision_ps_plane(std::string _name, float _rest, float _sratio, float _)
	QMap<QString, object*>& objects() { return objs; }
	QMap<QString, collision*>& collisions() { return cs; }
	//tObject objectTypeFromName(QString nm);
	object* objectFromStr(QString& str) { return objs[str]; }
	unsigned int numPolygonSphere();
	unsigned int numParticle() { return ps ? ps->numParticle() : 0; }
	unsigned int numPlane() { return planes.size(); }
	unsigned int numCube() { return cubes.size(); }
	unsigned int numCylinder() { return cylinders.size(); }
	unsigned int numPolygonObject() { return pObjs.size(); }
	unsigned int numCollision() { return cs.size(); }
	unsigned int numMass() { return masses.size(); }
	QString& modelPath() { return model_path; }
	QString& modelName() { return name; }

	particle_system* particleSystem() { return ps; }
	QMap<QString, collision*>& collision_map() { return cs; }
	QMap<QString, polygonObject>& objPolygon() { return pObjs; }
	QMap<QString, kinematicConstraint*>& kinConstraint() { return consts; }
	QMap<object*, mass*>& pointMasses() { return masses; }
	QMap<QString, cylinder>& objCylinders() { return cylinders; }

	VEC3F gravity() { return grav; }

	//QFile& modelStream() { return io_model; }
	//QFile& mphysicsStream() { return io_mph; }
	//void closeStream() { io_model.close(); io_mph.close(); }

	void saveModeler();
	void openModeler(GLWidget *gl, QString& file);
	//void updateObject(float dt, tSolveDevice tsd = CPU);
	void runExpression(float ct, float dt);

private:
	VEC3F grav;
	tUnit unit;								// unit
	tGravity dg;							// direction of gravity

	QString model_path;
	QString name;
	tSimulation tsim;
	QMap<QString, object*> objs;
	QMap<object*, mass*> masses;
	QMap<QString, cube> cubes;
	QMap<QString, cylinder> cylinders;
	QMap<QString, plane> planes;
	QMap<QString, polygonObject> pObjs;
	QMap<QString, collision*> cs;

	QMap<QString, kinematicConstraint*> consts;

	particle_system *ps;
};

#endif