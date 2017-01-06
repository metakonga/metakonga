#include "parSIM.h"
//#include "Simulation.h"
using namespace parSIM;
using namespace parSIM::mass;

rigid_body::rigid_body(Simulation *sim, std::string _name, geometry* _Geo, mass_type mt)
	: pointmass(sim, _name, _Geo, mt)
{

}

rigid_body::~rigid_body()
{

}

void rigid_body::define(unsigned int Id, double m, vector3<double>& diagIner, vector3<double>& symIner, vector3<double>& _pos, euler_parameter<double>& epara)
{
	pointmass::id = Id;
	pointmass::mass = m;
	pointmass::prin_iner = diagIner;
	pointmass::sym_iner = symIner;
	pointmass::pos = _pos;
	//pointmass::vel.x = -1.0;
	pointmass::ep = epara;
	pointmass::inertia.a00 = diagIner.x;
	pointmass::inertia.a11 = diagIner.y;
	pointmass::inertia.a22 = diagIner.z;
	define_mass();
	
	if(Geo && Geo->Geometry() == SHAPE){
		Geo->bindPointMass(this);
		geo::shape *sh = dynamic_cast<geo::shape*>(Geo);
		sh->update_polygons();
	}
}

void rigid_body::save2file(std::fstream& of)
{
	//std::cout << "position y is " << pos.y << std::endl;
// 	if (pos.y < 0.03574 && forceCalculator::OnGravityForce == true)
// 	{
// 		forceCalculator::OnGravityForce = false;
// 		pointmass::OnMoving = true;
// 		Simulation::cStep = 0;
// 		std::cout << "-------------------------------------------------------------------------" << std::endl;
// 		std::cout << "---------------------- Moving Condition is begin!! ----------------------" << std::endl;
// 		std::cout << "-------------------------------------------------------------------------" << std::endl;
// 		std::cout << "position y is " << pos.y << std::endl;
// 		vel.x = 0.0; vel.y = 0.0; vel.z = 0.0;
// 	}
		
	int name_size = name.size();
	of.write((char*)&name_size, sizeof(int));
	of.write((char*)name.c_str(), sizeof(char) * name_size);
	of.write((char*)&pos, sizeof(vector3<double>));
	of.write((char*)&vel, sizeof(vector3<double>));
	of.write((char*)&force, sizeof(vector3<double>));
}

void rigid_body::define_mass()
{
	sim->insert_pointmass(pointmass::name, this);
	MakeTransformationMatrix();
// 	if(sim->Device() == GPU){
// 		define_device_info();
// 	}
}