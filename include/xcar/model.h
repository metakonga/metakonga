#ifndef MODEL_H
#define MODEL_H

#include <QString>
#include "vectorTypes.h"
#include "types.h"
#include "resultStorage.h"

class model
{
public:
	model();
	~model();

	static void setModelName(QString& n) { name = n; }
	static void setModelPath(QString& p) { path = p; }
	static void setGravity(double g, gravity_direction d);
	//static void 

	static bool isSinglePrecision;
	static unit_type unit;
	static QString name;
	static QString path;
	static VEC3D gravity;
	static resultStorage *rs;
	static int count;
};

#endif