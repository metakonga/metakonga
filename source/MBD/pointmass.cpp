#include "parSIM.h"
#include "cu_dem_dec.cuh"

using namespace parSIM;
bool pointmass::OnMoving = false;

pointmass::pointmass( Simulation* _sim, std::string _name, geometry* _Geo, mass_type mt)
	: sim(_sim)
	, name(_name)
	, Geo(_Geo)
	, m_type(mt)
	, d_A(NULL)
	, d_pos(NULL)
	, isMovingPart(false)
{

}

pointmass::~pointmass()
{
	if(d_A) checkCudaErrors( cudaFree(d_A) ); d_A = NULL;
	if(d_pos) checkCudaErrors( cudaFree(d_pos) ); d_pos = NULL;
}

vector3<double> pointmass::toLocal(vector3<double> &v)
{
	vector3<double> tv;
	tv.x = A.a00*v.x + A.a10*v.y + A.a20*v.z;
	tv.y = A.a01*v.x + A.a11*v.y + A.a21*v.z;
	tv.z = A.a02*v.x + A.a12*v.y + A.a22*v.z;
	return tv;
}

vector3<double> pointmass::toGlobal(vector3<double> &v)
{
	vector3<double> tv;
	tv.x = A.a00*v.x + A.a01*v.y + A.a02*v.z;
	tv.y = A.a10*v.x + A.a11*v.y + A.a12*v.z;
	tv.z = A.a20*v.x + A.a21*v.y + A.a22*v.z;
	return tv;
}

void pointmass::setInertia()
{
	inertia.diagonal(POINTER3(prin_iner));
}

void pointmass::MakeTransformationMatrix()
{
	A.a00=2*(ep.e0*ep.e0+ep.e1*ep.e1-0.5);	A.a01=2*(ep.e1*ep.e2-ep.e0*ep.e3);		A.a02=2*(ep.e1*ep.e3+ep.e0*ep.e2);
	A.a10=2*(ep.e1*ep.e2+ep.e0*ep.e3);		A.a11=2*(ep.e0*ep.e0+ep.e2*ep.e2-0.5);	A.a12=2*(ep.e2*ep.e3-ep.e0*ep.e1);
	A.a20=2*(ep.e1*ep.e3-ep.e0*ep.e2);		A.a21=2*(ep.e2*ep.e3+ep.e0*ep.e1);		A.a22=2*(ep.e0*ep.e0+ep.e3*ep.e3-0.5);
}

void pointmass::update_geometry_data()
{
	if(!Geo)
		return;
	//vector4<double> *pos = sim->getParticles()->Position();
	
	geo::shape *g = dynamic_cast<geo::shape*>(Geo);
	Geo->Position() = pos;
	//std::fstream pf;
	//pf.open("E:/dem_debug_data/shape_vertice_info.txt", std::ios::out);
	
	//pf.close();
	g->update_polygons();
	g->body_force = vector3<double>(0);
	g->plane_body_force = vector3<double>(0);
	g->body_moment = vector4<double>(0);
}

void pointmass::cu_update_geometry_data()
{
	if(!Geo)
		return;
	if(!d_A){
		define_device_info();
	}
	cu_mass_info mi;
	geo::shape *g = dynamic_cast<geo::shape*>(Geo);
	mi.pos = make_double3(pos.x, pos.y, pos.z);
	mi.vel = make_double3(vel.x, vel.y, vel.z);
	checkCudaErrors( cudaMemcpy(g->d_mass_info, &mi, sizeof(cu_mass_info), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_pos, &pos, sizeof(double3), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_A, &A.a00, sizeof(double) * 9, cudaMemcpyHostToDevice) );	
	
	Geo->cu_update_geometry(d_A, d_pos);

}

void pointmass::define_device_info()
{
	checkCudaErrors( cudaMalloc((void**)&d_A, sizeof(double) * 9) );
	checkCudaErrors( cudaMalloc((void**)&d_pos, sizeof(double3)) );

	checkCudaErrors( cudaMemcpy(d_A, &A.a00, sizeof(double) * 9, cudaMemcpyHostToDevice) );
}

void pointmass::setMovingFunction(vector3<double>(*func)(vector3<double>, double))
{
	isMovingPart = true;
	MovingFunc = func;
}

void pointmass::RunMoving(double t)
{
	if (OnMoving)
		pos = MovingFunc(pos, t);
}
