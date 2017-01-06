#include "parRock_v2/RockSimulation.hpp"

typedef float base_type;

vector3<base_type> func1(base_type t)
{
	return t * vector3<base_type>(0.0f, -1.0f, 0.0);
}

int main(int argc, char** argv)
{
	Simulation<float> *sim = new RockSimulation<base_type>(DIM_2, "C:/C++/result/", "pickandrock2");
	sim->SetSpecificData("C:/C++/result/assembly.dat");
	//sim->FloatingDataType() = 's';
	RockSimulationParameters<base_type>::tm_req_isostr_tol = 0.5f;
	RockSimulationParameters<base_type>::tm_req_isostr = -1.0e6f;
	RockSimulationParameters<unsigned int>::flt_def = 3;
	RockSimulationParameters<base_type>::flt_r_mult = 1.1f;
	RockSimulationParameters<base_type>::f_tol = 0.1f;
	RockSimulationParameters<base_type>::relax = 1.5f;
	RockSimulationParameters<base_type>::hyst = 0.9f;
	RockSimulationParameters<bool>::densityScaling = true;

	WallElement<base_type>::wYoungsFactor = 1.2f;
	WallElement<base_type>::wfriction = 0.3f;

	RockElement<base_type>::maxDiameter = 0.000913f;
	RockElement<base_type>::diameterRatio = 1.66f;
	RockElement<base_type>::porosity = 0.16f;
	RockElement<base_type>::density = 2630.f;
	RockElement<base_type>::ryoungsModulus = 62e+9f;
	RockElement<base_type>::poissonRatio = 0.3f;
	RockElement<base_type>::rstiffnessRatio = 2.5f;
	RockElement<base_type>::friction = 0.3f;

	CementElement<base_type>::brmul = 1.f;
	CementElement<base_type>::cyoungsModulus = 62e+9f;
	CementElement<base_type>::cstiffnessRatio = 2.5f;
	CementElement<base_type>::maxTensileStress = 157e+6f;
	CementElement<base_type>::maxShearStress = 175e+6f;
	CementElement<base_type>::tensileStdDeviation = 36e+6f;
	CementElement<base_type>::shearStdDeviation = 40e+6f;

	Contact<base_type>::contact_stiffness_model = 'l';

	Geometry<base_type>* specimen = sim->CreateGeometry(geometry_shape::RECTANGLE, "specimen", geometry_type::GEO_PARTICLE);
	specimen->Define(vector3<float>(0.0f, 0.0f, 0.0f), vector3<float>(0.0317f, 0.0634f, 0.0f));
	//specimen->Define(vector3<float>(0.0f, 0.0f, 0.0f), vector3<float>(0.005f, 0.005f, 0.0f));

	Geometry<base_type>* boundary = sim->CreateGeometry(geometry_shape::RECTANGLE, "boundary", geometry_type::GEO_BOUNDARY);
	boundary->Define(vector3<base_type>(0.0f, 0.0f, 0.0f), vector3<base_type>(0.0317f, 0.0634f, 0.0f));

	sim->SaveGeometries('b');

	if (!sim->Initialize()){
		std::cout << "ERROR : Initialize is failed." << std::endl;
	}
	
	boundary->IsContact() = false;

	Geometry<base_type>* BottomLine = sim->CreateGeometry(geometry_shape::LINE, "BottomLine", geometry_type::GEO_BOUNDARY);
	BottomLine->Define(vector3<base_type>(0.0f, 0.0f, 0.0f), vector3<base_type>(0.0317f, 0.0f, 0.0f), vector3<base_type>(0.0f, 1.0f, 0.0));

	Geometry<base_type>* UpperLine = sim->CreateGeometry(geometry_shape::LINE, "UpperLine", geometry_type::GEO_BOUNDARY);
	UpperLine->Define(vector3<base_type>(0.0f, 0.06345f, 0.0f), vector3<base_type>(0.0317f, 0.06345f, 0.0f), vector3<base_type>(0.0f, -1.0f, 0.0f));
	UpperLine->setFunction(func1);

	sim->RunCycle(1);
	//specimen->Define()
	//sim->SetSorter();
	/*geo::Rectangle *specimen = new geo::Rectangle(sim, "specimen", GEO_PARTICLE);
	specimen->Define(vector3<float>(0.0f, 0.0f, 0.0f), vector3<float>(0.0634f, 0.0317f, 0.0f));
	
	utility::writer::SaveGeometry();

	if (!sim->Initialize()){
		std::cout << "ERROR : Initialize is failed." << std::endl;
	}*/
	delete sim;
	return 0;
}