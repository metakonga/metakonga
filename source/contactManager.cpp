#include "contactManager.h"
#include "particleManager.h"
#include "geometryObjects.h"
#include "database.h"
#include "contact_particles_particles.h"
#include "contact_particles_cube.h"
#include "contact_particles_plane.h"
#include "contact_particles_polygonObject.h"

contactManager::contactManager()
{

}

contactManager::~contactManager()
{
	qDeleteAll(cots);
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
	double rest, ratio, fric;
	qts >> ch;
	while (ch != "END_DATA")
	{
		qts >> _name
			>> ch >> method
			>> ch >> obj0
			>> ch >> obj1
			>> ch >> rest
			>> ch >> ratio
			>> ch >> fric;
		object* o1 = obj0 == "particles" ? pm->Object() : objs->Object(obj0);
		object* o2 = obj1 == "particles" ? pm->Object() : objs->Object(obj1);
		CreateContactPair(
			_name, method, o1, o2, rest, ratio, fric);
		qts >> ch;
	}
}

void contactManager::insertContact(contact* c)
{
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
	
	foreach(contact *c, cots)
	{
		c->collision(
			pos, vel, omega, 
			mass, force, moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	return true;
}

void contactManager::CreateContactPair(
	QString n, int method, object* o1, object* o2, 
	double rest, double ratio, double fric)
{
	contact::pairType pt = contact::getContactPair(o1->ObjectType(), o2->ObjectType());
	contact *c = NULL;
	switch (pt)
	{
	case contact::PARTICLE_PARTICLE:
		c = new contact_particles_particles(n, (contactForce_type)method, o1, o2);
		break;
	case contact::PARTICLE_CUBE:
		c = new contact_particles_cube(n, (contactForce_type)method, o1, o2);
		break;
	case contact::PARTICLE_PANE:
		c = new contact_particles_plane(n, (contactForce_type)method, o1, o2);
		break;
	}
	material_property_pair mpp =
	{
		o1->Youngs(), o2->Youngs(),
		o1->Poisson(), o2->Poisson(),
		o1->Shear(), o2->Shear()
	};
	
	c->setMaterialPair(mpp);
	c->setContactParameters(rest, ratio, fric);
	this->insertContact(c);
	database::DB()->addChild(database::COLLISION_ROOT, c->Name());

	QString log;
	QTextStream qts(&log);
	qts << "NAME " << n << endl
		<< "METHOD " << method << endl
		<< "OBJECT0 " << o1->Name() << endl
		<< "OBJECT1 " << o2->Name() << endl
		<< "RESTITUTION " << rest << endl
		<< "RATIO " << ratio << endl
		<< "FRICTION " << fric << endl;
	logs[c->Name()] = log;
}

void contactManager::CreateParticlePolygonsPairs(
	QString n, int method, object* po, QMap<int, polygonObject*>& pobjs,
	double rest, double ratio, double fric)
{
	contact_particles_polygonObject *c = new contact_particles_polygonObject(n, (contactForce_type)method, po, &pobjs);
	unsigned int nPolySphere = 0;
	foreach(polygonObject* pobj, pobjs)
	{
		c->insertContactParameters(pobj->Number(), rest, ratio, fric);
		nPolySphere += pobj->NumTriangle();
	}
	c->allocPolygonInformation(nPolySphere);
	nPolySphere = 0;
	foreach(polygonObject* pobj, pobjs)
	{
		c->definePolygonInformation(pobj->Number(), nPolySphere, pobj->NumTriangle(), pobj->VertexList(), pobj->IndexList());
		nPolySphere += pobj->NumTriangle();
	}
	cppoly = c;
}
