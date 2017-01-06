#ifndef VPOLYGON_H
#define VPOLYGON_H

#include "vglew.h"
#include "mphysics_numeric.h"

class QTextStream;

class vpolygon : public vglew
{
public:
	vpolygon();
	vpolygon(QString& _name);
	//vpolygon(QTextStream& in);
	~vpolygon();

	void draw(GLenum eMode);

	bool define(VEC3D org, host_polygon_info* hpi, VEC4D* _sphere, VEC3D* vset, VEC3UI* iset, unsigned int nid, unsigned int nvt);
	//bool makePolygonGeometry(QTextStream& in);
	bool makePolygonGeometry(VEC3F& P, VEC3F& Q, VEC3F& R);
	void setResultData(unsigned int nout);
	void insertResultData(unsigned int i, VEC3D& p, EPD& r);

private:
	unsigned int nvertex;
	unsigned int nindex;
	unsigned int m_vertex_vbo;
	unsigned int m_index_vbo;
	float origin[3];
	float ang[3];
	VEC3F *vertice;
	VEC3F *normals;
	VEC3UI *indice;
	VEC4F *spheres;

	VEC3D* outPos;
	EPD* outRot;
};

#endif