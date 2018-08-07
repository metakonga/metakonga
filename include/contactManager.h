#ifndef CONTACTMANAGER_H
#define CONTACTMANAGER_H

#include <QMap>
#include <QTextStream>
#include "contact.h"
#include "object.h"

class geometryObjects;
class particleManager;
class polygonObject;
class contact_particles_polygonObject;

class contactManager
{
public:
	contactManager();
	~contactManager();

	void Save(QTextStream& qts);
	void Open(QTextStream& qts, particleManager* pm, geometryObjects* objs);
	void CreateContactPair(
		QString n, int method, object* fo, object* so, 
		double rest, double ratio, double fric);
	void CreateParticlePolygonsPairs(
		QString n, int method, object* po, QMap<int, polygonObject*>& pobjs,
		double rest, double ratio, double fric);
	void insertContact(contact* c);
	contact* Contact(QString n);// { return cots[n]; }
	QMap<QString, QString>& Logs() { return logs; }
	QMap<QString, contact*>& Contacts() { return cots; }
	contact_particles_polygonObject* ContactPoly() { return cppoly; }
	bool runCollision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

private:
	QMap<QString, QString> logs;
	QMap<QString, contact*> cots;

	contact_particles_polygonObject* cppoly;
};

#endif