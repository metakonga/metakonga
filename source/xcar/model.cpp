#include "model.h"

QString model::name = "Model1";
QString model::path = kor(getenv("USERPROFILE")) + "/Documents/";
VEC3D model::gravity = VEC3D(0.0, -9.80665, 0.0);
resultStorage* model::rs = NULL;// new resultStorage;
int model::count = -1;
unit_type model::unit = MKS;
bool model::isSinglePrecision = false;

model::model()
{
	count++;
	if (!rs)
		rs = new resultStorage;
}

model::~model()
{
	count--;
	if (rs) delete rs; rs = NULL;
}

void model::setGravity(double g, gravity_direction d)
{
	VEC3D u;
	switch (d)
	{
	case PLUS_X: u.x = 1.0; break;
	case PLUS_Y: u.y = 1.0; break;
	case PLUS_Z: u.z = 1.0; break;
	case MINUS_X: u.x = -1.0; break;
	case MINUS_Y: u.y = -1.0; break;
	case MINUS_Z: u.z = -1.0; break;
	}
	gravity = g * u;
}