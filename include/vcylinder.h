#ifndef VCYLINDER_H
#define VCYLINDER_H

#include "vobject.h"
#include <QTextStream>

class vcylinder : public vobject
{
public:
	vcylinder();
	vcylinder(QString& _name);
	vcylinder(QTextStream& in);
	virtual ~vcylinder(){ glDeleteLists(glList, 1); }

	virtual void draw(GLenum eMode);

	float eulerAngle_1() { return ang[0]; }
	float eulerAngle_2() { return ang[1]; }
	float eulerAngle_3() { return ang[2]; }

	bool define();
	bool makeCylinderGeometry(QTextStream& in);
	bool makeCylinderGeometry(float br, float tr, float leng, VEC3D org, VEC3D p1, VEC3D p2);

private:
	void setIndexList();

	unsigned int glList;
	float baseRadius;
	float topRadius;
	float length;
	float origin[3];
	float pb[3];
	float pt[3];
	float ang[3];
};

#endif