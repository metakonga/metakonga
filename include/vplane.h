#ifndef VPLANE_H
#define VPLANE_H

#include "vobject.h"

class QTextStream;

class vplane : public vobject
{
public:
	vplane();
	vplane(QString& _name);
	vplane(QTextStream& in);
	virtual ~vplane() { glDeleteLists(glList, 1); }

	virtual void draw(GLenum eMode);

	bool define();
	bool makePlaneGeometry(QTextStream& in);
	bool makePlaneGeometry(VEC3D& pa, VEC3D& pb, VEC3D& pc, VEC3D& pd);

private:
	unsigned int glHiList;
	unsigned int glList;
	float p0[3];
	float p1[3];
	float p2[3];
	float p3[3];
};

#endif