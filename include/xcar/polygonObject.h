#ifndef POLYGONOBJECT_H
#define POLYGONOBJECT_H

#include "pointMass.h"

class polygonObject : public pointMass
{
public:
	polygonObject();
	polygonObject(QString file, geometry_use _roll);
	//polygonObject(QString file);
	polygonObject(const polygonObject& _poly);
	virtual ~polygonObject();

	//virtual unsigned int makeParticles(double rad, VEC3UI &_size, VEC3D& spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos = NULL, unsigned int sid = 0);
	//virtual void cuAllocData(unsigned int _np);
	bool define(import_shape_type t, VEC3D& loc, int ntriangle, double* vList, unsigned int *iList);
	void updateDeviceFromHost();

// 	VEC3D Vertex0(unsigned int i) { return vertice[indice[i].x]; }
// 	VEC3D Vertex1(unsigned int i) { return vertice[indice[i].y]; }
// 	VEC3D Vertex2(unsigned int i) { return vertice[indice[i].z]; }

	QString meshDataFile() const { return filePath; }
// 	unsigned int numVertex() const { return nvertex; }
// 	unsigned int numIndex() const { return nindex; }
	double maxRadius() const { return maxRadii; }
// 	VEC3D* vertexSet() const { return vertice; }
// 	VEC3D* normalSet() const { return normals; }
// 	VEC3UI* indexSet() const { return indice; }
//	host_polygon_mass_info* hostMassInfo() const { return h_mass; }
	unsigned int NumTriangle() const { return ntriangle; }
	double* VertexList() { return vertexList; }
	unsigned int* IndexList() { return indexList; }
	//device_polygon_mass_info* deviceMassInfo() const { return d_mass; }
	//VEC4D* hostSphereSet() const { return h_sph; }
//	double* deviceSphereSet() const { return d_sph; }
///	host_polygon_info* hostPolygonInfo() const { return h_poly; }
	//device_polygon_info* devicePolygonInfo() const { return d_poly; }

	static unsigned int Number() { return nPolygonObject; }

private:
	void _fromMS3DASCII(int _ntriangle, double* vList, unsigned int *iList);
	void _fromSTLASCII(int _ntriangle, double* vList, VEC3D& loc);

private:
	static unsigned int nPolygonObject;
//	unsigned int nvertex;
//	unsigned int nindex;
	unsigned int ntriangle;
	double maxRadii;
	double *vertexList;
	unsigned int *indexList;
//	unsigned int *id;
	//VEC3D *vertice;
	//VEC3UI *indice;
	//VEC3D *normals;
	//VEC4D *h_sph;
	//VEC3D *local_vertice;
	//double *d_sph;
	//host_polygon_info* h_poly;
//	device_polygon_info* d_poly;
//	host_polygon_mass_info* h_mass;
//	device_polygon_mass_info* d_mass;

	QString filePath;
};

#endif