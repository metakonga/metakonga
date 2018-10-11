#ifndef CONTACTMANAGER_H
#define CONTACTMANAGER_H

#include <QMap>
#include <QTextStream>
#include "contact.h"
#include "object.h"
#include "contact_particles_particles.h"

class geometryObjects;
class particleManager;
class polygonObject;
class contact_particles_polygonObject;
/*class contact_particles_particles;*/
class contact_particles_polygonObjects;

class contactManager
{
public:
	contactManager();
	~contactManager();

	void Save(QTextStream& qts);
	void Open(QTextStream& qts, particleManager* pm, geometryObjects* objs);
	void CreateContactPair(
		QString n, int method, object* fo, object* so, 
		double rest, double ratio, double fric, double coh);
// 	contact* CreateParticlePolygonsPairs(
// 		QString n, int method, object* po, QMap<int, polygonObject*>& pobjs,
// 		double rest, double ratio, double fric);
	unsigned int setupParticlesPolygonObjectsContact();
	double* SphereData();
	void insertContact(contact* c);
	contact* Contact(QString n);// { return cots[n]; }
	QMap<QString, QString>& Logs() { return logs; }
	QMap<QString, contact*>& Contacts() { return cots; }
	contact_particles_particles* ContactParticles() { return cpp; }
	contact_particles_polygonObjects* ContactParticlesPolygonObjects() { return cppoly; }
	bool runCollision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	void update();

private:
	void deviceCollision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	void hostCollision(
		VEC4D *pos, VEC3D *vel,
		VEC3D *omega, double *mass,
		VEC3D *force, VEC3D *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	QMap<QString, QString> logs;
	QMap<QString, contact*> cots;
	QMap<QString, contact_particles_polygonObject*> cppos;

	contact_particles_particles* cpp;
	contact_particles_polygonObjects* cppoly;
};

#endif