#ifndef CUBE_H
#define CUBE_H

#include "pointMass.h"
#include "plane.h"

class cube : public pointMass
{
public:
	cube(){}
	cube(QString& _name, geometry_use _roll);
	cube(const cube& _cube);
	virtual ~cube();

	plane* Planes() const { return planes; }
	device_plane_info* deviceCubeInfo() { return dpi; }

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

	static unsigned int Number() { return nCube; }

private:
	static unsigned int nCube;
	VEC3D ori;
	VEC3D min_p;
	VEC3D max_p;
	VEC3D size;
	plane *planes;

	device_plane_info *dpi;
};

#endif