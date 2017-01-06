#ifndef CUBE_H
#define CUBE_H

#include "object.h"
#include "plane.h"

// QT_BEGIN_NAMESPACE
// class QTextStream;
// QT_END_NAMESPACE

class cube : public object
{
public:
	cube(){}
	cube(modeler* _md, QString& _name, tMaterial _mat, tRoll _roll);
	cube(const cube& _cube);
	~cube();

	virtual unsigned int makeParticles(float rad, float spacing, bool isOnlyCount, VEC4F_PTR pos = NULL, unsigned int sid = 0);
	virtual void cuAllocData(unsigned int _np){}
	virtual void updateMotion(float t, tSolveDevice tsd){}
	virtual void updateFromMass(){}

	void save_shape_data(QTextStream& ts) const;

	bool define(vector3<float>& min, vector3<float>& max);
	vector3<float> origin() { return ori; }
	vector3<float> origin() const { return ori; }
	vector3<float> min_point() { return min_p; }
	vector3<float> min_point() const { return min_p; }
	vector3<float> max_point() { return max_p; }
	vector3<float> max_point() const { return max_p; }
	vector3<float> cube_size() { return size; }
	vector3<float> cube_size() const { return size; }
	plane planes_data(int i) const { return planes[i]; }

private:
	vector3<float> ori;
	vector3<float> min_p;
	vector3<float> max_p;
	vector3<float> size;
	plane planes[6];
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