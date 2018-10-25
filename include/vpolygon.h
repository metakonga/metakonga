#ifndef VPOLYGON_H
#define VPOLYGON_H

#include "vglew.h"
#include "vectorTypes.h"
#include "types.h"
#include "vobject.h"

#include <QList>

class QTextStream;

class vpolygon : public vobject// , public vglew
{
public:
	vpolygon();
	vpolygon(QString& _name);
	//vpolygon(QTextStream& in);
	virtual ~vpolygon();

	virtual void draw(GLenum eMode);
	
	bool define(import_shape_type t, QString file);
	bool makePolygonGeometry(VEC3F& P, VEC3F& Q, VEC3F& R);
	void setResultData(unsigned int nout);
	void insertResultData(unsigned int i, VEC3D& p, EPD& r);
	void splitTriangle(double to);
	QList<triangle_info> _splitTriangle(triangle_info& ti, double to);

	//unsigned int ID() { return id; }
//	QString name() { return nm; }
	unsigned int NumTriangles() { return ntriangle; }
	double* VertexList() { return vertexList; }
	float* VertexList_f() { return vertice; }
	unsigned int* IndexList() { return indexList; }
	void setSelected(bool b) { isSelected = b; }

private:
	void _drawPolygons();
	void _loadMS3DASCII(QString f);
	void _loadSTLASCII(QString f);
	
	//unsigned int glHList;

	unsigned int nvertex;
	unsigned int nvtriangle;
	unsigned int ntriangle;
	unsigned int m_vertex_vbo;
	unsigned int m_index_vbo;
	unsigned int m_color_vbo;
	unsigned int m_normal_vbo;
	
	float origin[3];
	float ang[3];
	float *vertice;
	double *vertexList;
//	float *vertexList_f;
	unsigned int *indexList;
	float *normals;
	float *texture;
	unsigned int *indice;
	float *colors;
	VEC4F *spheres;
	VEC3D min_point;
	VEC3D max_point;
	vobject* select_cube;
// 	VEC3D* outPos;
// 	EPD* outRot;

	shaderProgram program;
};

#endif