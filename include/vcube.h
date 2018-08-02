#ifndef VCUBE_H
#define VCUBE_H

#include "vobject.h"

class QTextStream;

class vcube : public vobject
{
public:
	vcube();
	vcube(QString& _name);
	vcube(QTextStream& in);
	virtual ~vcube(){ glDeleteLists(glList, 1); }

	virtual void draw(GLenum eMode);

	bool define();
	bool makeCubeGeometry(QTextStream& in);
	bool makeCubeGeometry(QString& _name, geometry_use _tr, material_type _tm, VEC3F& _mp, VEC3F& _sz);

private:
	void setIndexList();
	void setNormalList();
	unsigned int glList;
	unsigned int glHiList;
	float origin[3];
	int indice[24];
	float vertice[24];
	float normal[18];
};

#endif