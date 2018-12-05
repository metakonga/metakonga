#ifndef DEM_MODEL_H
#define DEM_MODEL_H

#include "particleManager.h"
#include "grid_base.h"
#include "dem_integrator.h"
#include "model.h"
#include <QMap>
#include <QFile>
#include <QTextStream>
#include <QString>

class dem_model : public model
{
public:
	dem_model();
	~dem_model();

	particleManager* ParticleManager() { return pm; }
	grid_base::Type SortType() { return sort_type; }
	dem_integrator::Type IntegrationType() { return integration_type; }

	void setSortType(grid_base::Type t) { sort_type = t; }
	void setIntegrationType(dem_integrator::Type t) { integration_type = t; }

	particleManager* CreateParticleManager();
	void Save(QTextStream& qts);
	void Open(QTextStream& qts);

	void setRollingConditionEnable(bool rce) { rollingCondition = rce; }
	bool RollingCondition() { return rollingCondition; }
	//void setParticleSystem(particleManager* _ps) { ps = _ps; }
// 	void setDatabase(database* _db) { db = _db; }
// 	void setSolveDevice(tSolveDevice tsd) { tdevice = tsd; }
//	mass* makeMass(QString _name);
	
// 	kinematicConstraint* makeKinematicConstraint(QString _name, tKinematicConstraint kt, mass* i, VEC3D& spi, VEC3D& fi, VEC3D& gi, mass* j, VEC3D& spj, VEC3D& fj, VEC3D& gj);
// 	drivingConstraint* makeDrivingConstraint(QString _name, kinematicConstraint* kconst, tDriving td, double val);
	
//	particleManager* makeParticleSystem(QString _name);
//	collision* makeCollision(QString _name, double _rest, double _fric, double _rfric, double _coh, double _ratio, tCollisionPair tcp, tContactModel tcm, void *o1, void *o2 = NULL);
// 	QMap<QString, object*>& objects() { return objs; }
// 	QMap<QString, collision*>& collisions() { return cs; }
// 	object* objectFromStr(QString& str) { return objs[str]; }
// 	unsigned int numPolygonSphere();
// 	unsigned int numParticle() { return ps ? ps->numParticle() : 0; }
// 	unsigned int numCollision() { return cs.size(); }
// 	unsigned int numMass() { return masses.size(); }
// 	QString& modelPath() { return model_path; }
// 	QString& modelName() { return name; }
// 	void actionDelete(const QString& tg);
// 	particle_system* particleSystem() { return ps; }
// 	QList<polygonObject*>& polyObjects() { return polygons; }
// 	QMap<QString, collision*>& collision_map() { return cs; }
// 	QMap<QString, kinematicConstraint*>& kinConstraint() { return consts; }
// 	QMap<QString, drivingConstraint*>& drivingConstraints() { return dconsts; }
// 	QMap<object*, mass*>& pointMasses() { return masses; }
// 	
// 	VEC3D gravity() { return grav; }

	//void saveModeler();
	//void openModeler(GLWidget *gl, QString& file);
	//void updateObject(float dt, tSolveDevice tsd = CPU);
	//void runExpression(double ct, double dt);
	//void runDriving(double ct);
// 	template<typename object_type>
// 	object_type getChildObject(const QString& str)
// 	{
// 		return dynamic_cast<object_type>(objs.find(str).value());
// 	}

private:
	bool rollingCondition;
	dem_integrator::Type integration_type;
	grid_base::Type sort_type;
	particleManager *pm;
};

#endif