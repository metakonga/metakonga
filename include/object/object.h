#ifndef OBJECT_H
#define OBJECT_H

#include "mphysics_types.h"
#include "mphysics_numeric.h"
#include <QFile>
#include <QString>

QT_BEGIN_NAMESPACE
class QTextStream;
QT_END_NAMESPACE

class modeler;
class mass;

class object
{
public:
	object();
	object(modeler *_md, QString& _name, tObject _tobj, tMaterial _mat, tRoll _roll);
	object(const object& obj);
	virtual ~object();

	virtual unsigned int makeParticles(float rad, float spacing, bool isOnlyCount, VEC4F_PTR pos = NULL, unsigned int sid = 0) = 0;
	virtual void cuAllocData(unsigned int _np) = 0;
	virtual void updateMotion(float t, tSolveDevice tsd = CPU) = 0;
	virtual void updateFromMass() = 0;

	unsigned int ID() const { return id; }
	//virtual void saveResult(std::fstream& of) = 0;
	
	QString objectName() const { return name; }
	//void setObjectName(std::string& _name) { name = _name; }
	tObject objectType() const { return obj_type; }
	tMaterial materialType() const { return mat_type; }
	tRoll rolltype() const { return roll_type; }
	bool expression() const { return _expression; }

	//void setObjectName(QString& _n) { name = _n; }
	void setRoll(tRoll tr)  { roll_type = tr; }
	void setUpdate(bool b) { _expression = b; }
	void setID(unsigned int _id) { sid = _id > sid ? _id : sid; id = _id; }
	void setMaterial(tMaterial _tm);

	float density() const { return d; }
	float youngs() const { return y; }
	float poisson() const { return p; }
	float shear() const { return sm; }
	bool isUpdate() const { return _update; }
	mass* pointMass() const { return ms; }
	void addPointMass(mass* _ms) { ms = _ms; }
	void save_mass_data(QTextStream& ts) const;
	void setRelativeImpactVelocity(unsigned int _np) { riv = new float[_np]; }
	void setCudaRelativeImpactVelocity(unsigned int _np);
	float* relativeImpactVelocity() { return riv; }
	float* cuRelativeImpactVelocity() { return d_riv; }

	void runExpression(float ct, float dt);
	//static void close_stream() { io_object.close(); }

protected:
	static unsigned int sid;
	unsigned int id;

	bool _expression;
	//static std::ofstream io_object;
	bool _update;
	QString name;
	tRoll roll_type;
	tObject obj_type;
	tMaterial mat_type;
	float d;		// density
	float y;		// young's modulus
	float p;		// poisson ratio
	float sm;		// shear modulus
	//simulation *sim;
	float *riv;
	float *d_riv;
	mass* ms;
	
	modeler* md;
};

#endif