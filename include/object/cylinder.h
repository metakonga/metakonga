#ifndef CYLINDER_H
#define CYLINDER_H

#include "object.h"

struct device_cylinder_info;

class cylinder : public object
{
public:
	cylinder();
	cylinder(modeler* _md, QString& _name, tMaterial _mat, tRoll _roll);
	cylinder(const cylinder& _cube);
	virtual ~cylinder();

	virtual unsigned int makeParticles(double rad, VEC3UI &_size, VEC3D &spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np);
	virtual void updateMotion(double t, tSolveDevice tsd){}
	virtual void updateFromMass();

	virtual void save_object_data(QTextStream& ts);
	device_cylinder_info* deviceCylinderInfo() { return dci; }

	bool define(double _br, double _tr, VEC3D _bpos, VEC3D _tpos);
	double baseRadisu() const { return br; }
	double topRadius() const { return tr; }
	double length() const { return len; }
	VEC3D origin() const { return org; }
	//VEC3F dirPosition() const { return dpos; }
	VEC3D basePos() const { return bpos; }
	VEC3D topPos() const { return tpos; }
	void setOrientation(double e1, double e2, double e3);
	EPD orientation() const { return ep; }
// 	EPD& t_orientation() { return t_ep; }
// 	EPD& b_orientation() { return b_ep; }
//	VEC3F sidePos1() const { return spos1; }
//	VEC3F sidePos2() const { return spos2; }

private:
	double br;				// base radius
	double tr;				// top radius
	double len;				// length
	VEC3D org;		// origin
	EPD ep;
// 	EPD b_ep;
// 	EPD t_ep;
	//vector3<double> dpos;
	VEC3D bpos;
	VEC3D tpos;
	VEC3D loc_bpos;
	VEC3D loc_tpos;

	device_cylinder_info* dci;
	//MAT33F A;
	//VEC3F spos1;
	//VEC3F spos2;
};

// inline
// std::ostream& operator<<(std::ostream& oss, const cube& my){
// 	oss << "OBJECT CUBE " << my.objectName() << " " << my.rolltype() << " " << my.materialType() << std::endl;
// 	oss << my.origin() << std::endl
// 		<< my.min_point() << std::endl
// 		<< my.max_point() << std::endl
// 		<< my.cube_size() << std::endl;
// 	return oss;
// }

#endif