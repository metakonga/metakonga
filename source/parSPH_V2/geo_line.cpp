#include "geo_line.h"
#include "sphydrodynamics.h"
#include <cmath>

using namespace geo;

line::line(sphydrodynamics* _sph, tParticle _tp, std::string _nm)
	: geometry(_sph, _tp, LINE, _nm)
{

}

line::~line()
{

}

void line::define(VEC3F& start, VEC3F& end, bool normalStartEndLeft, bool considerHP, bool _isInner)
{
	startPoint = start;
	endPoint = end;
	normal = (end - start).rotate2((((int)normalStartEndLeft) * 2 - 1)*(float)M_PI_2).normalize();
	considerHP = considerHP;
	isInner = _isInner;
}

void line::build(bool onlyCountParticles)
{
	pcount = 0;

	VEC3F diff = endPoint - startPoint;
	int lineCnt = (int)(diff.length() / sph->particleSpacing() + 0.5f);
	float spacing = diff.length() / lineCnt;
	VEC3F unitDiff = diff.normalize();

	InitParticle(startPoint, normal, unitDiff, onlyCountParticles, true, 0, false, isInner);

	for (int i = 1; i < lineCnt; i++){
		VEC3F displacement = startPoint + (i * spacing) * unitDiff;
		InitParticle(displacement, normal, unitDiff, onlyCountParticles, false, 0, false, isInner);
	}

	InitParticle(endPoint, normal, unitDiff, onlyCountParticles, true, 0, false, isInner);
}

bool line::particleCollision(const VEC3F& position, float radius)
{
	return utils::circleLineIntersect(startPoint, endPoint, position, radius);
}

std::vector<corner> line::corners()
{
	corner c1 = { startPoint, normal, (startPoint - endPoint).normalize() };
	corner c2 = { endPoint, normal, (endPoint - startPoint).normalize() };
	std::vector<corner> list(2);
	list[0] = c1;
	list[1] = c2;
	return list;
}

void line::initExpression()
{
// 	fluid_particle *fp = sph->particle(sid);
// 	for (unsigned int j = 0; j < pcount; j++){
// 		if (fp[j].particleType() == DUMMY)
// 			continue;
// 		fp[j].setVelocity(initVel);
// 		fp[j].setAuxVelocity(initVel);
// 	}
// 	for (unsigned int i = 0; i < pcount; i++){
// 		if (fp[i].particleType() == DUMMY){
// 			fp[i].setVelocity(initVel);
// 			fp[i].setAuxVelocity(initVel);
// 		}
// 	}
}