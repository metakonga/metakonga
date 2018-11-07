#include "contactManager.h"
#include "particleManager.h"
#include "geometryObjects.h"
#include "database.h"
/*#include "contact_particles_particles.h"*/
#include "contact_particles_cube.h"
#include "contact_particles_plane.h"
#include "contact_particles_polygonObject.h"
#include "contact_particles_polygonObjects.h"
#include "model.h"
#include "contact_plane_polygonObject.h"
#include <QDebug>

contactManager::contactManager()
	: cppoly(NULL)
	, cpp(NULL)
{

}

contactManager::~contactManager()
{
	qDeleteAll(cots);
	qDeleteAll(cppos);
	if (cpp) delete cpp; cpp = NULL;
	if (cppoly) delete cppoly; cppoly = NULL;
}

void contactManager::Save(QTextStream& qts)
{
	qts << endl
		<< "CONTACT_ELEMENTS_DATA" << endl;
	foreach(QString log, logs)
	{
		qts << log;
	}
	qts << "END_DATA" << endl;
}

void contactManager::Open(QTextStream& qts, particleManager* pm, geometryObjects* objs)
{
	QString ch;
	QString _name, obj0, obj1;
	int method;
	double rest, ratio, fric, coh;
	qts >> ch;
	while (ch != "END_DATA")
	{
		qts >> _name
			>> ch >> method
			>> ch >> obj0
			>> ch >> obj1
			>> ch >> rest
			>> ch >> ratio
			>> ch >> fric
			>> ch >> coh;
		object* o1 = obj0 == "particles" ? pm->Object() : objs->Object(obj0);
		object* o2 = obj1 == "particles" ? pm->Object() : objs->Object(obj1);
		CreateContactPair(
			_name, method, o1, o2, rest, ratio, fric, coh);
		qts >> ch;
	}
}

void contactManager::insertContact(contact* c)
{
	if (c->PairType() == contact::PARTICLE_POLYGON_SHAPE)
		cppos[c->Name()] = dynamic_cast<contact_particles_polygonObject*>(c);
	else
		cots[c->Name()] = c;
}

contact* contactManager::Contact(QString n)
{
	QStringList l = cots.keys();
	QStringList::const_iterator it = qFind(l, n);
	if (it == l.end())
		return NULL;
	return cots[n];
}

bool contactManager::runCollision(
	double *pos, double *vel, 
	double *omega, double *mass, 
	double *force, double *moment, 
	unsigned int *sorted_id, 
	unsigned int *cell_start,
	unsigned int *cell_end,
	unsigned int np)
{
	if (simulation::isCpu())
	{
		hostCollision
			(
			(VEC4D*)pos,
			(VEC3D*)vel,
			(VEC3D*)omega,
			mass,
			(VEC3D*)force,
			(VEC3D*)moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	else if (simulation::isGpu())
	{
		deviceCollision(
			pos, vel, omega,
			mass, force, moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	return true;
}

bool contactManager::runCollision(
	float *pos, float *vel, float *omega, 
	float *mass, float *force, float *moment, 
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
	if (simulation::isCpu())
	{
// 		hostCollision
// 			(
// 			(VEC4D*)pos,
// 			(VEC3D*)vel,
// 			(VEC3D*)omega,
// 			mass,
// 			(VEC3D*)force,
// 			(VEC3D*)moment,
// 			sorted_id, cell_start, cell_end, np
// 			);
	}
	else if (simulation::isGpu())
	{
		deviceCollision(
			pos, vel, omega,
			mass, force, moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	return true;
}

void contactManager::update()
{
	if (cppoly)
	{
		model::isSinglePrecision ?
			cppoly->updatePolygonObjectData_f() :
			cppoly->updatePolygonObjectData();
	}
		
}

void contactManager::deviceCollision(
	double *pos, double *vel, 
	double *omega, double *mass, 
	double *force, double *moment, 
	unsigned int *sorted_id, unsigned int *cell_start, 
	unsigned int *cell_end, unsigned int np)
{
	if (cpp)
	{
		cpp->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
	foreach(contact* c, cots)
	{
		if (c->IgnoreTime() && (simulation::ctime > c->IgnoreTime()))
			continue;
		if (c->IsEnabled())
			c->cuda_collision(
				pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
	if (cppoly)
	{
		//qDebug() << "pass_cuda_collision_cppoly0";
		cppoly->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
		//qDebug() << "pass_cuda_collision_cppoly1";
	}
// 	foreach(polygonObject* pobj, pair_ip)
// 	{
// 
// 	}
}
// 
void contactManager::deviceCollision(float *pos, float *vel, float *omega, float *mass, float *force, float *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	if (cpp)
	{
		cpp->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
	foreach(contact* c, cots)
	{
		if (c->IgnoreTime() && (simulation::ctime > c->IgnoreTime()))
			continue;
		if (c->IsEnabled())
			c->cuda_collision(
			pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
	if (cppoly)
	{
		cppoly->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
		
	}
}

void contactManager::hostCollision(
	VEC4D *pos, VEC3D *vel, VEC3D *omega, 
	double *mass, VEC3D *force, VEC3D *moment, 
	unsigned int *sorted_id, unsigned int *cell_start, 
	unsigned int *cell_end, unsigned int np)
{
	if (cppoly)
	{
		cppoly->setNumContact(0);
		cppoly->setZeroCollisionForce();
	}
		
	for (unsigned int i = 0; i < np; i++)
	{
		VEC3D F, M;
		VEC3D p = VEC3D(pos[i].x, pos[i].y, pos[i].z);
		VEC3D v = vel[i];
		VEC3D o = omega[i];
		double m = mass[i];
		double r = pos[i].w;
		foreach(contact* c, cots)
		{
			if (c->IgnoreTime() && (simulation::ctime > c->IgnoreTime()))
				continue;
			if (c->IsEnabled())
				c->collision(r, m, p, v, o, F, M);
		}
		VEC3I gp = grid_base::getCellNumber(p.x, p.y, p.z);
		for (int z = -1; z <= 1; z++){
			for (int y = -1; y <= 1; y++){
				for (int x = -1; x <= 1; x++){
					VEC3I neigh(gp.x + x, gp.y + y, gp.z + z);
					unsigned int hash = grid_base::getHash(neigh);
					unsigned int sid = cell_start[hash];
					if (sid != 0xffffffff){
						unsigned int eid = cell_end[hash];
						for (unsigned int j = sid; j < eid; j++){
							unsigned int k = sorted_id[j];
							if (i != k && k < np)
							{
								VEC3D jp(pos[k].x, pos[k].y, pos[k].z);
								VEC3D jv = vel[k];
								VEC3D jo = omega[k];
								double jr = pos[k].w;
								double jm = mass[k];
								cpp->cppCollision(
									r, jr,
									m, jm,
									p, jp,
									v, jv,
									o, jo,
									F, M);
							}
							else if (k >= np)
							{
								if (!cppoly->cppolyCollision(k - np, r, m, p, v, o, F, M))
								{

								}
							}
						}
					}
				}
			}
		}
		force[i] += F;
		moment[i] += M;             
	}
	//qDebug() << cppoly->NumContact();
}

void contactManager::CreateContactPair(
	QString n, int method, object* o1, object* o2, 
	double rest, double ratio, double fric, double cohesion)
{
	contact::pairType pt = contact::getContactPair(o1->ObjectType(), o2->ObjectType());
	contact *c = NULL;
	switch (pt)
	{
	case contact::PARTICLE_PARTICLE:
		cpp = new contact_particles_particles(n, (contactForce_type)method, o1, o2);	
		c = cpp;
		break;
	case contact::PARTICLE_CUBE:
		c = new contact_particles_cube(n, (contactForce_type)method, o1, o2);
		cots[c->Name()] = c;// this->insertContact(c);
		break;
	case contact::PARTICLE_PANE:
		c = new contact_particles_plane(n, (contactForce_type)method, o1, o2);
		cots[c->Name()] = c;// this->insertContact(c);
		break;
	case contact::PARTICLE_POLYGON_SHAPE:
		c = new contact_particles_polygonObject(n, (contactForce_type)method, o1, o2);
		cppos[c->Name()] = dynamic_cast<contact_particles_polygonObject*>(c);
		break;
	case contact::PLANE_POLYGON_SHAPE:
		c = new contact_plane_polygonObject(n, (contactForce_type)method, o1, o2);
		cots[c->Name()] = c;
		break;
	}
	material_property_pair mpp =
	{
		o1->Youngs(), o2->Youngs(),
		o1->Poisson(), o2->Poisson(),
		o1->Shear(), o2->Shear()
	};
	
	c->setMaterialPair(mpp);
	c->setContactParameters(rest, ratio, fric, cohesion);
	database::DB()->addChild(database::COLLISION_ROOT, c->Name());

	QString log;
	QTextStream qts(&log);
	qts << "NAME " << n << endl
		<< "METHOD " << method << endl
		<< "OBJECT0 " << o1->Name() << endl
		<< "OBJECT1 " << o2->Name() << endl
		<< "RESTITUTION " << rest << endl
		<< "RATIO " << ratio << endl
		<< "FRICTION " << fric << endl
		<< "COHESION " << cohesion << endl;
	logs[c->Name()] = log;
}

unsigned int contactManager::setupParticlesPolygonObjectsContact()
{
	unsigned int n = 0;
	if (cppos.size() && !cppoly)
	{
		cppoly = new contact_particles_polygonObjects;
		if (model::isSinglePrecision)
			n = cppoly->define_f(cppos);
		else
			n = cppoly->define(cppos);
	}	
	return n;
}

double* contactManager::SphereData()
{
	if (cppoly)
		return simulation::isGpu() ? cppoly->SphereData() : (double*)cppoly->HostSphereData();
	return NULL;
}

double* contactManager::HostSphereData()
{
	if (cppoly)
		return (double*)cppoly->HostSphereData();
	return NULL;
}

float* contactManager::SphereData_f()
{
	if (cppoly)
		return cppoly->SphereData_f();
	return NULL;
}

// contact* contactManager::CreateParticlePolygonsPairs(
// 	QString n, int method, object* po, QMap<int, polygonObject*>& pobjs,
// 	double rest, double ratio, double fric)
// {
// 	contact_particles_polygonObject *c = new contact_particles_polygonObject(n, (contactForce_type)method, po, &pobjs);
// 	unsigned int nPolySphere = 0;
// 	foreach(polygonObject* pobj, pobjs)
// 	{
// 		c->insertContactParameters(pobj->Number(), rest, ratio, fric);
// 		nPolySphere += pobj->NumTriangle();
// 	}
// 	c->allocPolygonInformation(nPolySphere);
// 	nPolySphere = 0;
// 	foreach(polygonObject* pobj, pobjs)
// 	{
// 		c->definePolygonInformation(pobj->Number(), nPolySphere, pobj->NumTriangle(), pobj->VertexList(), pobj->IndexList());
// 		nPolySphere += pobj->NumTriangle();
// 	}
// 	cppoly = c;
// 	return c;
// }
