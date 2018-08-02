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
	bool makePlaneGeometry(float l1, VEC3F& xw, VEC3F& pa, VEC3F& p2, VEC3F& u1);

private:
	unsigned int glHiList;
	unsigned int glList;
	float p0[3];
	float p1[3];
	float p2[3];
	float p3[3];
};

#endif