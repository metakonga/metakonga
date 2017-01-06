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

	virtual unsigned int makeParticles(float rad, float spacing, bool isOnlyCount, VEC4F_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np);
	virtual void updateMotion(float t, tSolveDevice tsd){}
	virtual void updateFromMass();

	void save_shape_data(QTextStream& ts) const;
	device_cylinder_info* deviceCylinderInfo() { return dci; }

	bool define(float _br, float _tr, vector3<float> _bpos, VEC3F _tpos);
	double baseRadisu() const { return br; }
	double topRadius() const { return tr; }
	double length() const { return len; }
	vector3<double> origin() const { return org; }
	//VEC3F dirPosition() const { return dpos; }
	VEC3D basePos() const { return bpos; }
	VEC3D topPos() const { return tpos; }
	void setOrientation(float e1, float e2, float e3);
	EPD orientation() const { return ep; }
	EPD& t_orientation() { return t_ep; }
	EPD& b_orientation() { return b_ep; }
//	VEC3F sidePos1() const { return spos1; }
//	VEC3F sidePos2() const { return spos2; }

private:
	double br;				// base radius
	double tr;				// top radius
	double len;				// length
	vector3<double> org;		// origin
	EPD ep;
	EPD b_ep;
	EPD t_ep;
	//vector3<float> dpos;
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