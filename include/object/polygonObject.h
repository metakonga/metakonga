#ifndef POLYGON_H
#define POLYGON_H

#include "object.h"

QT_BEGIN_NAMESPACE
class QTextStream;
QT_END_NAMESPACE

struct device_polygon_info;
struct device_polygon_mass_info;
struct double4;

class polygonObject : public object
{
public:
	polygonObject();
	polygonObject(modeler *_md, QString file);
	polygonObject(const polygonObject& _poly);
	virtual ~polygonObject();

	virtual unsigned int makeParticles(float rad, float spacing, bool isOnlyCount, VEC4F_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np);
	virtual void updateMotion(float t, tSolveDevice tsd){}
	virtual void updateFromMass();
	void updateDeviceFromHost();

	void save_shape_data(QTextStream& ts) const;
	bool define(tImport tm, QTextStream& file);

	QString meshDataFile() const { return filePath; }
	unsigned int numVertex() const { return nvertex; }
	unsigned int numIndex() const { return nindex; }
	double maxRadius() const { return maxRadii; }
	VEC3D* vertexSet() const { return vertice; }
	VEC3UI* indexSet() const { return indice; }
	host_polygon_mass_info* hostMassInfo() const { return h_mass; }
	device_polygon_mass_info* deviceMassInfo() const { return d_mass; }
	VEC4D* hostSphereSet() const { return h_sph; }
	double* deviceSphereSet() const { return d_sph; }
	host_polygon_info* hostPolygonInfo() const { return h_poly; }
	device_polygon_info* devicePolygonInfo() const { return d_poly; }
	VEC3D getOrigin() const { return org; }

private:
	void fromMS3DASCII(QTextStream& file);

private:
	unsigned int nvertex;
	unsigned int nindex;
	double maxRadii;
	VEC3D org;
	VEC3D *vertice;
	VEC3UI *indice;
	VEC4D *h_sph;
	double *d_sph;
	host_polygon_info* h_poly;
	device_polygon_info* d_poly;
	host_polygon_mass_info* h_mass;
	device_polygon_mass_info* d_mass;

	QString filePath;
};

#endif