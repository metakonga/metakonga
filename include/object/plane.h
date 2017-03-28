#ifndef PLANE_H
#define PLANE_H

#include "object.h"

QT_BEGIN_NAMESPACE
class QTextStream;
QT_END_NAMESPACE

struct device_plane_info;

class plane : public object
{
public:
	plane();
	plane(modeler *_md, QString& _name, tMaterial _mat, tRoll _roll);
	plane(const plane& _plane);
	~plane();

	virtual unsigned int makeParticles(double _rad, VEC3UI &_size, VEC3D& spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np);
	virtual void updateMotion(double t, tSolveDevice tsd);
	virtual void updateFromMass(){}

	virtual void save_object_data(QTextStream& ts);

	device_plane_info* devicePlaneInfo() { return dpi; }

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

private:
	double l1, l2;
	VEC3D u1;
	VEC3D u2;
	VEC3D uw;
	VEC3D xw;
	VEC3D pa;
	VEC3D pb;

	VEC3D w2;
	VEC3D w3;
	VEC3D w4;

	device_plane_info *dpi;
};

// inline
// std::ostream& operator<<(std::ostream& oss, const plane& my)
// {
// 	oss << "OBJECT PLANE " << my.objectName() << " " << my.rolltype() << " " << my.materialType() << std::endl;
// 	oss << my.L1() << std::endl
// 		<< my.L2() << std::endl
// 		<< my.U1() << std::endl
// 		<< my.U2() << std::endl
// 		<< my.UW() << std::endl
// 		<< my.XW() << std::endl
// 		<< my.PA() << std::endl
// 		<< my.PB() << std::endl;
// 
// 	return oss;
// }

#endif