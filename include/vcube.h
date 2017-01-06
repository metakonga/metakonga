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
	bool makeCubeGeometry(QString& _name, tRoll _tr, tMaterial _tm, VEC3F& _mp, VEC3F& _sz);

private:
	void setIndexList();

	unsigned int glList;
	int indice[24];
	float vertice[24];
};

#endif