#ifndef PLANE_H
#define PLANE_H

#include "pointMass.h"

struct device_plane_info;

class plane : public pointMass
{
public:
	plane();
	plane(QString& _name, geometry_use _roll);
	plane(const plane& _plane);
	virtual ~plane();

	bool define(VEC3D& _xw, VEC3D& _pa, VEC3D& _pc, VEC3D& _pb);
	bool define(VEC3D& _xw, VEC3D& _pa, VEC3D& _pb);
	double L1() const { return l1; }
	double L2() const { return l2; }
	VEC3D U1() const { return u1; }
	VEC3D U2() const { return u2; }
	VEC3D UW() const { return uw; }
	VEC3D XW() const { return xw; }
	VEC3D PA() const { return pa; }
	VEC3D PB() const { return pb; }
	VEC3D W2() const { return w2; }
	VEC3D W3() const { return w3; }
	VEC3D W4() const { return w4; }

	static unsigned int Number() { return nPlane; }

private:
	static unsigned int nPlane;
	double l1, l2;
	VEC3D minp;
	VEC3D maxp;
	VEC3D u1;
	VEC3D u2;
	VEC3D uw;
	VEC3D xw;
	VEC3D pa;
	VEC3D pb;

	VEC3D w2;
	VEC3D w3;
	VEC3D w4;
};

#endif