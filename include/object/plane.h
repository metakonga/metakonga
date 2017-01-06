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

	virtual unsigned int makeParticles(float _rad, float spacing, bool isOnlyCount, VEC4F_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np);
	virtual void updateMotion(float t, tSolveDevice tsd);
	virtual void updateFromMass(){}

	void save_shape_data(QTextStream& ts) const;

	device_plane_info* devicePlaneInfo() { return dpi; }

	bool define(vector3<float>& _xw, vector3<float>& _pa, vector3<float>& _pc, vector3<float>& _pb);
	bool define(vector3<float>& _xw, vector3<float>& _pa, vector3<float>& _pb);
	float L1() const { return l1; }
	float L2() const { return l2; }
	vector3<float> U1() const { return u1; }
	vector3<float> U2() const { return u2; }
	vector3<float> UW() const { return uw; }
	vector3<float> XW() const { return xw; }
	vector3<float> PA() const { return pa; }
	vector3<float> PB() const { return pb; }
	vector3<float> W2() const { return w2; }
	vector3<float> W3() const { return w3; }
	vector3<float> W4() const { return w4; }

private:
	float l1, l2;
	vector3<float> u1;
	vector3<float> u2;
	vector3<float> uw;
	vector3<float> xw;
	vector3<float> pa;
	vector3<float> pb;

	vector3<float> w2;
	vector3<float> w3;
	vector3<float> w4;

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