#ifndef GEOMETRYOBJECTS_H
#define GEOMETRYOBJECTS_H

#include "object.h"

class cube;
class plane;
class cylinder;
class polygonObject;
class pointMass;

class geometryObjects
{
public:
	geometryObjects();
	~geometryObjects();

	void Save(QTextStream& qts);
	void Open(QTextStream& qts);
	void insertObject(object* _obj);
	QMap<QString, object*>& Objects() { return objs; }
	QMap<QString, QString>& Logs() { return logs; }
	object* Object(QString n);// { return objs[n]; }
	QString polyRefinement(QString n, double v);
	void addMarkerObject(object* o){ markers.push_back(o); }
	cube* makeCube(
		QString _name, material_type mt, geometry_use gu, 
		VEC3D& start, VEC3D& end, double e, double p, double d, double s);
	plane* makePlane(
		QString _name, material_type mt, geometry_use gu, 
		VEC3D& pa, VEC3D& pb, VEC3D& pc, VEC3D& pd, 
		double e, double p, double d, double s);
	cylinder* makeCylinder(QString _name, double e, double p, double d, geometry_use _roll);
	polygonObject* makePolygonObject
		(QString _nm, geometry_use gu, QString file, import_shape_type t, VEC3D& loc, unsigned int ntri, double* vertexList, 
		unsigned int* indexList ,material_type mt, double e, double p, double d, double s);

private:
	QMap<QString, QString> logs;
	QMap<QString, object*> objs;
	QList<object*> markers;
	QMap<int, polygonObject*> pobjs;
};

#endif