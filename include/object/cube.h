#ifndef CUBE_H
#define CUBE_H

#include "object.h"
#include "plane.h"

class cube : public object
{
public:
	cube(){}
	cube(modeler* _md, QString& _name, tMaterial _mat, tRoll _roll);
	cube(const cube& _cube);
	virtual ~cube();

	virtual unsigned int makeParticles(double rad, VEC3UI &_size, VEC3D &spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np){}
	virtual void updateMotion(double t, tSolveDevice tsd){}
	virtual void updateFromMass(){}

	virtual void save_object_data(QTextStream& ts);

	bool define(VEC3D& min, VEC3D& max);
	VEC3D origin() { return ori; }
	VEC3D origin() const { return ori; }
	VEC3D min_point() { return min_p; }
	VEC3D min_point() const { return min_p; }
	VEC3D max_point() { return max_p; }
	VEC3D max_point() const { return max_p; }
	VEC3D cube_size() { return size; }
	VEC3D cube_size() const { return size; }
	plane planes_data(int i) const { return planes[i]; }

private:
	VEC3D ori;
	VEC3D min_p;
	VEC3D max_p;
	VEC3D size;
	plane planes[6];
};

#endif