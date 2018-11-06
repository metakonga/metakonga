#ifndef MODELMANAGER_H
#define MODELMANAGER_H

#include "geometryObjects.h"
#include "dem_model.h"
#include "mbd_model.h"
//#include "database.h"
#include "particleManager.h"
#include "contactManager.h"

//#include "FullCarModel.hpp"
//#include "SliderCrank3D.hpp"
//#include "test_model.hpp"

class modelManager
{
public:
	enum modelType{DEM, MBD, OBJECTS, PARTICLE_MANAGER, CONTACT_MANAGER, ALL};
	modelManager();
	~modelManager();

	static modelManager* MM();

	void SaveCurrentModel();
	void OpenModel(QString file_path);
	void ActionDelete(QString target);
	void CreateModel(QString& n, modelType t, bool isOnAir = false);
	//void CreateContactPair(QString n, int method, QString fo, QString so, double rest, double ratio, double fric);

	QMap<QString, geometryObjects*>& GeometryObjects() { return objs; }
	QMap<QString, dem_model*>& DEMModels() { return dems; }
	QMap<QString, mbd_model*>& MBDModels() { return mbds; }
	QMap<QString, contactManager*>& ContactManagers() { return contacts; }
	//QMap<QString, particleManager*>& ParticleManagers() { return pms; }

	geometryObjects* GeometryObject(QString& n);// { return objs[n]; }
	dem_model* DEMModel(QString& n);// { return dems[n]; }
	mbd_model* MBDModel(QString& n);// { return mbds[n]; }
	contactManager* ContactManager(QString& n);
	//particleManager* ParticleManager(QString& n);

	geometryObjects* GeometryObject() { return obj; }
	dem_model* DEMModel() { return dem; }
	mbd_model* MBDModel() { return mbd; }
//	particleManager* ParticleManager() { return pm; }
	contactManager* ContactManager() { return cont; }

	void setOnAirModel(modelType t, QString& n);
	//void setDatabase(database* _db) { db = _db; }
// 
// 	bool defineFullCarModel();
// 	void defineSliderCrank3D();
// 	bool defineTestModel();

private:
	QMap<QString, geometryObjects*> objs;
	QMap<QString, dem_model*> dems;
	QMap<QString, mbd_model*> mbds;
	QMap<QString, contactManager*> contacts;
	//QMap<QString, particleManager*> pms;

	geometryObjects* obj;
	dem_model* dem;
	mbd_model* mbd;
	contactManager* cont;
	//particleManager *pm;

//	database* db;
};

#endif