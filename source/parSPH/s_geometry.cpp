#include "s_geometry.h"
#include "sphydrodynamics.h"
#include "s_particle.h"
#include "cu_sph_decl.cuh"
#include <cmath>
#include <cfloat>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace parsph;

unsigned int Geometry::objectCount = 0;

Geometry::Geometry(sphydrodynamics* parentSimulation, t_particle particleType, geometry_type gt)
	: startId(0)
	, particleCount(0)
	, sph(parentSimulation)
	, type(particleType)
	, isMovement(false)
	, startMovementTime(0)
	, endMovementTime(0)
	, considerHP(false)
	, gtype(gt)
{
	objectId = objectCount;
	name = "object_" + utils::integerString(objectId);
	sph->models.insert(std::pair<std::string, Geometry*>(name, this));
	objectCount++;
}

Geometry::Geometry(sphydrodynamics* parentSimulation, t_particle particleType, std::string name, geometry_type gt)
	: startId(0)
	, particleCount(0)
	, isMovement(false)
	, sph(parentSimulation)
	, type(particleType)
	, considerHP(false)
	, gtype(gt)
{
	objectId = objectCount;

	if(name.empty())
		this->name = "object_" + utils::integerString(objectId);
	else
		this->name = name;

	sph->models.insert(std::pair<std::string,Geometry*>(this->name, this));

	objectCount++;
}

Geometry::~Geometry()
{

}

unsigned int Geometry::ParticleCount()
{
 	if(!particleCount)
 		Build(true);
  	return particleCount;
}

void Geometry::InitParticle(const vector3<double>& pos, const vector3<double>& normal, bool onlyCountParticles, bool isCorner, int minusCount, bool isf)
{
	if(isCorner)
		if(sph->isCornerOverlapping(pos))
			return;

	if(!onlyCountParticles)
	{
		s_particle* p = sph->getParticle(startId + particleCount);
		p->setID(startId + particleCount);
		p->setType(BOUNDARY);
		p->setPosition(pos);
		p->setIsFloating(isf);
		p->setDensity(sph->Density());
		p->setMass(sph->particleMass[FLUID]);
		p->setPressure(0.);
		p->setVelocity(isMovement ? initVel : vector3<double>(0.0, 0.0, 0.0));
		p->setNormal(normal);
	}

	particleCount += 1 + sph->initDummies(startId + particleCount, pos, normal, onlyCountParticles, considerHP, minusCount, isf);
}

void Geometry::SetVelocity(vector3<double> velocity)
{
	/*if(particleCount)
	{
		sim->program->Argument("VECTOR_VALUE")->Write(velocity);
		sim->program->Argument("OBJECT_START")->Write(startId);
		sim->program->Argument("OBJECT_PARTICLE_COUNT")->Write(particleCount);
		sim->EnqueueSubprogram("upload attribute", Utils::NearestMultiple(particleCount, 512));
		}*/
}

void Geometry::SetInitExpression(std::string attribute, std::vector<std::string> expression)
{
	initializationExp.insert(std::pair<std::string,std::vector<std::string> >(attribute, expression));
}

void geo::Line::InitExpression()
{
	for(std::map<std::string, std::vector<std::string>>::iterator it = initializationExp.begin(); it != initializationExp.end(); it++){
		if(it->first == "VELOCITIES"){
			int i = 0;
			double vel[3] = {0, };
			for(std::vector<std::string>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++, i++){
				vel[i] = atof(it2->c_str());
			}
			s_particle *ps = sph->getParticle(startId);
			for(unsigned int j = 0; j < particleCount; j++){
				if(ps[j].Type() == DUMMY)
					continue;
				ps[j].setVelocity(vector3<double>(vel[0], vel[1], vel[2]));
			}
		}
	}
}

void Geometry::InitExpressionDummyParticles()
{
	if(!isMovement)
		return;
	s_particle *ps = sph->getParticle(startId);
	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() == DUMMY){
			ps[i].setVelocity(initVel);
		}
	}
}

void Geometry::SetMovementExpression(double startTime, double endTime, vector3<double> iniVel)
{
// 	positionExp = position;
// 	velocityExp = velocity;
	startMovementTime = startTime;
	endMovementTime = endTime;
	initVel = iniVel;
}

void Geometry::RunExpression(double dt, double time)
{
	if(sph->Device() == GPU){
		if(time < endMovementTime && time > startMovementTime){
			cu_runExpression(sph->d_pos + startId, sph->d_vel + startId, time, particleCount);
		}
		
	}
	s_particle *ps = sph->getParticle(startId);
	for(unsigned int j = 0; j < particleCount; j++){
		vector3<double> vel(0.1, 0.0, 0.0);
		ps[j].Position() += dt * vel;
		//ps[j].setVelocity(vel);
	}
}

void geo::Line::Define(vector2<double> start, vector2<double> end, bool normalStartEndLeft, bool cHydroPressure)
{
	startPoint = start;
	endPoint = end;
	normal = (end - start).rotate((((int)normalStartEndLeft)*2 - 1)*M_PI_2).normalize();
	considerHP = cHydroPressure;
}

bool geo::Line::particleCollision(const vector3<double>& position, double radius)
{
	return utils::circleLineIntersect(startPoint, endPoint, position, radius);
}

void geo::Line::Build(bool onlyCountParticles)
{
	particleCount = 0;

	vector2<double> diff = endPoint - startPoint;
	int lineCnt = (int)(diff.length() / sph->particleSpacing() + 0.5);
	double spacing = diff.length() / lineCnt;
	vector2<double> unitDiff = diff.normalize();

	InitParticle(startPoint, normal, onlyCountParticles, true, 0, false);

	for(int i = 1; i < lineCnt; i++){
		vector2<double> displacement = startPoint + unitDiff * (i * spacing);
		InitParticle(displacement, normal, onlyCountParticles, false, 0, false);
	}

	InitParticle(endPoint, normal, onlyCountParticles, true, 0, false);
}

void geo::Line::Export(std::fstream& pf)
{
	char v = 'l';
	pf.write(&v, sizeof(char));
	pf.write((char*)&startPoint, sizeof(double)*2);
	pf.write((char*)&endPoint, sizeof(double)*2);
}

std::vector<Geometry::Corner> geo::Line::Corners()
{
	Geometry::Corner c1 = {startPoint, normal, (startPoint - endPoint).normalize()};
	Geometry::Corner c2 = {  endPoint, normal, (endPoint - startPoint).normalize()};
	std::vector<Geometry::Corner> list(2);
	list[0] = c1;
	list[1] = c2;
 	return list;
}

void geo::Square::Define(vector3<double> cg, vector3<double> dim, vector4<double> orient, bool isf)
{
	cofmass = cg;
	dimension = dim;
	orientation = orient;
	MakeTransformationMatrix(&A.a00, &orientation.x);
	Geometry::isFloating = isf;
}

std::vector<Geometry::Corner> geo::Square::Corners()
{
	normal[0] = A * vector3<double>(0.0, -1.0, 0.0);
	edge[0] = vector3<double>(cofmass.x - 0.5 * dimension.x, cofmass.y - 0.5 * dimension.y, cofmass.z - 0.5 * dimension.z);
	// 	Geometry::Corner c1 = { edge[0], normal[0]};
	// 	Geometry::Corner c2 = { }

	normal[1] = A * vector3<double>(-1.0, 0.0, 0.0);
	edge[1] = vector3<double>(cofmass.x - 0.5 * dimension.x, cofmass.y + 0.5 * dimension.y, cofmass.z - 0.5 * dimension.z);
	//Geometry::Corner c2 = {edge[1] , normal[1]};

	normal[2] = A * vector3<double>(0.0, 1.0, 0.0);
	edge[2] = vector3<double>(cofmass.x + 0.5 * dimension.x, cofmass.y + 0.5 * dimension.y, cofmass.z - 0.5 * dimension.z);
	//Geometry::Corner c3 = { edge[2], normal[2]};

	normal[3] = A * vector3<double>(1.0, 0.0, 0.0);
	edge[3] = vector3<double>(cofmass.x + 0.5 * dimension.x, cofmass.y - 0.5 * dimension.y, cofmass.z - 0.5 * dimension.z);
	//Geometry::Corner c4 = { edge[3], normal[3]};
	Geometry::Corner c1 = { edge[3], normal[0] };
	Geometry::Corner c2 = { edge[0], normal[0] };
	Geometry::Corner c3 = { edge[0], normal[1] };
	Geometry::Corner c4 = { edge[1], normal[1] };
	Geometry::Corner c5 = { edge[1], normal[2] };
	Geometry::Corner c6 = { edge[2], normal[2] };
	Geometry::Corner c7 = { edge[2], normal[3] };
	Geometry::Corner c8 = { edge[3], normal[3] };
	std::vector<Geometry::Corner> list(8);
	list[0] = c1;
	list[1] = c2;
	list[2] = c3;
	list[3] = c4;
	list[4] = c5;
	list[5] = c6;
	list[6] = c7;
	list[7] = c8;
	return list;
}

void geo::Square::Build(bool onlyCountParticles)
{
	particleCount = 0;
	vector3<double> diffs[4] = { edge[0] - edge[3],
		edge[1] - edge[0],
		edge[2] - edge[1],
		edge[3] - edge[2] };
	vector3<double> spoint[4] = { edge[3], edge[0], edge[1], edge[2] };
	vector3<double> epoint[4] = { edge[0], edge[1], edge[2], edge[3] };
	unsigned int layers = (unsigned int)(sph->GridCellSize() / sph->particleSpacing());
	if(onlyCountParticles)
		sph->SetNumInnerCornerParticles(layers * 4);
	int minusCount = 0;
	//int icount = 0;
	for(int i = 0; i < 4; i++){
		unsigned int lineCnt = (unsigned int)(diffs[i].length() / sph->particleSpacing() + 0.5);
		double spacing = diffs[i].length() / lineCnt;
		vector3<double> unitDiff = diffs[i].normalize();
		InitParticle(spoint[i], normal[i], onlyCountParticles, true, 0, Geometry::isFloating);
		for(unsigned int j = 1; j < lineCnt; j++){
			minusCount = 0;
			vector3<double> displacement = spoint[i] +  (j * spacing) * unitDiff;
			if(j <= layers)
			{
				minusCount = layers - j + 1;
				//icount = i 
			}
			else if(j >= lineCnt - layers)
			{
				minusCount = j - lineCnt + layers + 1;
			}
			InitParticle(displacement, normal[i], onlyCountParticles, false, minusCount, Geometry::isFloating);
		}
		InitParticle(epoint[i], normal[i], onlyCountParticles, true, 0, Geometry::isFloating);
	}
}

void geo::Square::InitExpression()
{

}

void geo::Square::Export(std::fstream& pf)
{

}