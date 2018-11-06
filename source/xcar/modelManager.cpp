#include "modelManager.h"
#include <QDir>

modelManager* mm;

modelManager::modelManager()
	: obj(NULL)
	, dem(NULL)
	, mbd(NULL)
	//, pm(NULL)
	, cont(NULL)
{
	mm = this;
}

modelManager::~modelManager()
{
	qDeleteAll(objs);
	qDeleteAll(mbds);
	qDeleteAll(dems);	
	qDeleteAll(contacts);
	//qDeleteAll(pms);
}

modelManager* modelManager::MM()
{
	return mm;
}

void modelManager::SaveCurrentModel()
{
	if (!QDir(model::path).exists())
		QDir().mkdir(model::path);
	QString file_name = model::path + model::name + ".xdm";
	QFile qf(file_name);
	qf.open(QIODevice::WriteOnly);
	QTextStream qts(&qf);
	qts << "MODEL_NAME " << model::name << endl;
	qts << "GRAVITY " << model::gravity.x << " " << model::gravity.y << " " << model::gravity.z << endl
		<< "UNIT " << (int)model::unit << endl
		<< "SINGLE " << model::isSinglePrecision << endl << endl;
	if (dem)
		dem->Save(qts);
	if (obj)
		obj->Save(qts);
	if (cont)
		cont->Save(qts);
	if (mbd)
		mbd->Save(qts);
	qf.close();
}

void modelManager::OpenModel(QString file_path)
{
	QFile qf(file_path);
	qf.open(QIODevice::ReadOnly);
	QTextStream qts(&qf);
	QString ch;
	while (!qts.atEnd())
	{
		qts >> ch;
		if (ch == "MODEL_NAME")
		{
			int unit;
			int iss;
			qts >> ch;
			model::setModelName(ch);
			qts >> ch >> model::gravity.x >> model::gravity.y >> model::gravity.z;
			qts >> ch >> unit;
			qts >> ch >> iss;
			model::isSinglePrecision = iss;
			model::unit = (unit_type)unit;
		}
		else if (ch == "DEM_MODEL_DATA")
		{
			//qts >> ch >> ch;
		//	model::setModelName(ch);
			CreateModel(model::name, DEM, true);
			dem->Open(qts);
		}
		else if (ch == "PARTICLES_DATA")
		{
			particleManager* pm = dem->CreateParticleManager();
			pm->Open(qts);
		}
		else if (ch == "GEOMETRY_OBJECTS_DATA")
		{
			CreateModel(model::name, OBJECTS, true);
			obj->Open(qts);
		}
		else if (ch == "CONTACT_ELEMENTS_DATA")
		{
			CreateModel(model::name, CONTACT_MANAGER, true);
			cont->Open(qts, dem->ParticleManager(), obj);
		}
		else if (ch == "MULTIBODY_MODEL_DATA")
		{
			qts >> ch;
			CreateModel(model::name, MBD, true);
// 			if (ch == "END_DATA")
// 				return;
			mbd->setMBDModelName(ch);
			mbd->Open(qts);
		}

	}
}

void modelManager::ActionDelete(QString target)
{
// 	object* o = obj->deleteObject(target);
// 	contact* c = cont->Contact(target);
	if (obj)
	{
		object* o = obj->Objects().take(target);// v_objs.take(tg);
		if (o) delete o;
	}	
	if (cont)
	{
		contact* c = cont->Contacts().take(target);
		if (c) delete c;
	}
			
}

void modelManager::CreateModel(QString& n, modelType t, bool isOnAir)
{
	switch (t)
	{
	case DEM: dems[n] = new dem_model; break;
	case MBD: mbds[n] = new mbd_model; break;
	case OBJECTS: objs[n] = new geometryObjects; break;
	case PARTICLE_MANAGER: dem->CreateParticleManager();
	case CONTACT_MANAGER: contacts[n] = new contactManager; break;
	}

	if (isOnAir)
	{
		setOnAirModel(t, n);
	}
}

geometryObjects* modelManager::GeometryObject(QString& n)
{
// 	QStringList s = objs.keys();
// 	QStringList::const_iterator it = qFind(s, n);
// 	if (it == NULL/*s.end()*/)
// 		return NULL;
	return objs[n];
}

dem_model* modelManager::DEMModel(QString& n)
{
// 	QStringList s = dems.keys();
// 	QStringList::const_iterator it = qFind(s, n);
// 	if (it == s.end())
// 		return NULL;
	return dems[n];
}

mbd_model* modelManager::MBDModel(QString& n)
{
// 	QStringList s = mbds.keys();
// 	QStringList::const_iterator it = qFind(s, n);
// 	if (it == s.end())
// 		return NULL;
	return mbds[n];
}

// particleManager* modelManager::ParticleManager()
// {
// 	return pms[n];
// }

contactManager* modelManager::ContactManager(QString& n)
{
	return contacts[n];
}

void modelManager::setOnAirModel(modelType t, QString& n)
{
	switch (t)
	{
	case DEM: dem = DEMModel(n); break;
	case MBD: mbd = MBDModel(n); break;
	case OBJECTS:obj = GeometryObject(n); break;
	//case PARTICLE_MANAGER: pm = ParticleManager(n); break;
	case CONTACT_MANAGER: cont = ContactManager(n); break;
	case ALL:
		dem = DEMModel(n);
		mbd = MBDModel(n);
		obj = GeometryObject(n);
	//	pm = ParticleManager(n);
		cont = ContactManager(n);
		break;
	}

}
// 
// bool modelManager::defineFullCarModel()
// {
// // 	FullCarModel* fcm = new FullCarModel();
// // 	bool ret = fcm->setUp();
// // 	if (ret)
// // 	{
// // 		mbds[model::name] = fcm;
// // 		mbd = fcm;
// // 	}
// // 	else
// // 	{
// // 		delete fcm;
// // 		return false;
// // 	}
// // 	
// // 	return ret;
// }
// 
// void modelManager::defineSliderCrank3D()
// {
// // 	SliderCrank3D* sc3d = new SliderCrank3D();
// // 	sc3d->setUp();
// // 	mbds[model::name] = sc3d;
// // 	mbd = sc3d;
// }
// 
// bool modelManager::defineTestModel()
// {
// // 	particle_tire_addition_model* tmd = new particle_tire_addition_model(mbd);
// // 	bool ret = tmd->setUp();
// // // 	if (ret)
// // // 	{
// // // 		mbds[model::name] = tmd;
// // // 		mbd = tmd;
// // // 	}
// // // 	else
// // // 	{
// // // 		delete tmd;
// // // 		return false;
// // // 	}
// // 	return ret;
// }
