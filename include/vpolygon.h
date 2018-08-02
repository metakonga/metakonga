#ifndef VPOLYGON_H
#define VPOLYGON_H

#include "vglew.h"
#include "vectorTypes.h"
#include "types.h"
#include "vobject.h"

class QTextStream;

class vpolygon : public vobject// , public vglew
{
public:
	vpolygon();
	vpolygon(QString& _name);
	//vpolygon(QTextStream& in);
	virtual ~vpolygon();

	virtual void draw(GLenum eMode);
	
	bool define(VEC3D org, VEC3D* nor, VEC4D* _sphere, VEC3D* vset, VEC3UI* iset, unsigned int nid, unsigned int nvt);
	bool makePolygonGeometry(VEC3F& P, VEC3F& Q, VEC3F& R);
	void setResultData(unsigned int nout);
	void insertResultData(unsigned int i, VEC3D& p, EPD& r);

	//unsigned int ID() { return id; }
//	QString name() { return nm; }
	void setSelected(bool b) { isSelected = b; }

private:
	void _drawPolygons();

	//QString nm;
	//bool isSelected;
	//static int pcnt;
	//unsigned int id;
	unsigned int nvertex;
	unsigned int nindex;
	unsigned int m_vertex_vbo;
	unsigned int m_index_vbo;
	unsigned int m_color_vbo;
	unsigned int m_normal_vbo;
	float origin[3];
	float ang[3];
	VEC3F *vertice;
	VEC3F *normals;
	VEC3UI *indice;
	VEC4F *colors;
	VEC4F *spheres;

// 	VEC3D* outPos;
// 	EPD* outRot;

	shaderProgram program;
};

#endif