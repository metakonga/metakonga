#include "RockSimulation.h"
#include <cmath>

RockSimulation::RockSimulation(std::string bpath, std::string cname)
	: Simulation(bpath, cname)
{

}

RockSimulation::~RockSimulation()
{

}

bool RockSimulation::DefineRockParticles()
{
	std::map<std::string, Geometry*>::iterator it = geometries.find("specimen");
	if (it == geometries.end())
	{
		std::cout << "Error : No exist the specimen geometry." << std::endl;
		return false;
	}
	if (RockElement::diameterRatio == 1.0f){

	}
	else{
		float minRadius = 0.5f * (RockElement::maxDiameter / RockElement::diameterRatio);
		float maxRadius = 0.5f * RockElement::maxDiameter;
		float ru = 0.5f * (maxRadius + minRadius);
		float area = 0.0f;
		switch (it->second->Shape()){
		case RECTANGLE:
		{
			geo::Rectangle *rec = dynamic_cast<geo::Rectangle*>(it->second);
			area = rec->Area();
			nball = static_cast<unsigned int>(area * (1 - RockElement::porosity) / (M_PI * ru * ru));
			if (!nball){
				std::cout << "Error : The number of ball is zero." << std::endl;
				return false;
			}
			else{
				std::cout << "The number of particle : " << nball << std::endl;
			}
			balls = new particles(nball);
			srand(1973);
			float Ap = 0.0f;
			for (unsigned int i = 0; i < balls->Np(); i++){
				float radii = 0.0f;
				while (radii <= minRadius){
					radii = maxRadius * ffrand();
				}
				balls->Radius(i) = 0.5f * radii;
				Ap += (float)M_PI * balls->Radius(i) * balls->Radius(i);
			}

			float n0 = (area - Ap) / area;
			float m = sqrt((1 - RockElement::porosity) / (1 - n0));
			for (unsigned int i = 0; i < nball; i++){
				//particle* b = &balls[i];
				balls->Radius(i) *= m;
				balls->Position()[i].x = rec->StartPoint().x + ffrand() * rec->Sizex();
				balls->Position()[i].y = rec->StartPoint().y + ffrand() * rec->Sizey();
				/*b->Acceleration().x = 0.0;
				//b->*///Acceleration().y = -9.80665;
				balls->Mass(i) = RockElement::density * (dim == DIM_2 ? balls->Radius(i) * balls->Radius(i) * (float)M_PI : (4.0f / 3.0f) * (float)M_PI * pow(balls->Radius(i), 3));
				balls->Inertia(i) = 2.0f * balls->Mass(i) * pow(balls->Radius(i), 2) / 5.0f;
			}
		}
		break;
		}
	}
	return true;
}

bool RockSimulation::Initialize()
{
	if (!DefineRockParticles()){
		std::cout << "ERROR : DefineRockParticle() function is returned the false." << std::endl;
		return false;
	}
	utility::writer::Save(0);
	return true;
}