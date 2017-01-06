#include "object.h"
#include "modeler.h"
#include "mass.h"
#include <cuda_runtime.h>

unsigned int object::sid = 0;
//std::ofstream object::io_object;

object::object()
	: riv(NULL)
	, d_riv(NULL)
	, ms(NULL)
	, _update(false)
{

}

object::object(modeler* _md, QString& _name, tObject _tobj, tMaterial _mat, tRoll _roll)
	: name(_name)
	, obj_type(_tobj)
	, mat_type(_mat)
	, roll_type(_roll)
	, _update(false)
	, _expression(false)
	, md(_md)
	, ms(NULL)
	, riv(NULL)
	, d_riv(NULL)
{
	id = sid++;
	d = material::getDensity(mat_type);
	y = material::getYoungs(mat_type);
	p = material::getPoisson(mat_type);
	sm = material::getShearModulus(mat_type);
}

object::object(const object& obj)
	: name(obj.objectName())
	, id(obj.ID())
	, obj_type(obj.objectType())
	, mat_type(obj.materialType())
	, roll_type(obj.rolltype())
	, d(obj.density())
	, y(obj.youngs())
	, p(obj.poisson())
	, sm(obj.shear())
	, _update(obj.isUpdate())
	, _expression(obj.expression())
	, ms(NULL)
	, riv(NULL)
	, d_riv(NULL)
{
	ms = obj.pointMass();
}

object::~object()
{
	//if (ms) delete ms; ms = NULL;
	if (riv) delete[] riv; riv = NULL;
	if (d_riv) cudaFree(d_riv); d_riv = NULL;

}

void object::save_mass_data(QTextStream& ts) const
{
	if(ms)
		ms->saveData(ts);
}

void object::setCudaRelativeImpactVelocity(unsigned int _np)
{
	cudaMalloc((void**)&d_riv, sizeof(float)*_np);
	cudaMemset(d_riv, 0, sizeof(float)*_np);
}

void object::setMaterial(tMaterial _tm)
{
	mat_type = _tm;
	d = material::getDensity(mat_type);
	y = material::getYoungs(mat_type);
	p = material::getPoisson(mat_type);
}

void object::runExpression(float ct, float dt)
{
	if (ct > 0.1){
		VEC3D am = /*((double)ct - 0.1) * */VEC3D(0.0, 0.0, 0.1);
		VEC3D omega = ms->getEV().toAngularVelocity(ms->getEP());
	//	double m_vel = VEC3D(0.05).cross(omega).length();
		//if (m_vel < 1.0)
		ms->setExternalMoment(am);
// 		else
// 			ms->setExternalMoment(VEC3D(0.0, 0.0, 0.0));
	}
}