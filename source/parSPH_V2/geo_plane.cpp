#include "geo_plane.h"
#include "sphydrodynamics.h"
#include <cmath>

using namespace geo;

plane::plane(sphydrodynamics* _sph, tParticle _tp, std::string _nm)
	: geometry(_sph, _tp, PLANE, _nm)
{

}

plane::~plane()
{

}

void plane::define(VEC3F& _p1, VEC3F& _p2, VEC3F& _p3, VEC3F& _p4, bool considerHP /* = false */, bool isInner /* = false */)
{
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	VEC3F vec_a = p2 - p1;
	VEC3F vec_b = p4 - p1;
	normal = vec_a.cross(vec_b).normalize();
}

void plane::build(bool onlyCountParticles)
{
 	pcount = 0;
	bool b1, b2, b3, b4;
	VEC3F vec_a = p2 - p1;
	VEC3F vec_b = p4 - p1;
 	int lineCnt_a = (int)(vec_a.length() / sph->particleSpacing() + 0.5f);
	int lineCnt_b = (int)(vec_b.length() / sph->particleSpacing() + 0.5f);
 	float spacing_a = vec_a.length() / lineCnt_a;
	float spacing_b = vec_b.length() / lineCnt_b;
	VEC3F unitdiff_a = vec_a.normalize();
	VEC3F unitdiff_b= vec_b.normalize();
	b1 = InitParticle(p1, normal, vec_a.normalize(), onlyCountParticles, true, 0, false, isInner);
	b2 = InitParticle(p2, normal, vec_a.normalize(), onlyCountParticles, true, 0, false, isInner);
	b3 = InitParticle(p3, normal, vec_b.normalize(), onlyCountParticles, true, 0, false, isInner);
	b4 = InitParticle(p4, normal, vec_b.normalize(), onlyCountParticles, true, 0, false, isInner);

	if (b1 && b4){
		if (onlyCountParticles)
		{
			overlappingLine ol = { 0, 0, p1, p4, -normal, normal.cross(unitdiff_b) };
			sph->insertOverlappingLines(ol);
		}
	}
	else{
		for (int i = 1; i < lineCnt_b; i++){
			VEC3F displacement = p1 + (i * spacing_b) * unitdiff_b;
			InitParticle(displacement, normal, unitdiff_a, onlyCountParticles, false, 0, false, isInner);
		}
	}

	for (int i = 1; i < lineCnt_a; i++){
		//VEC3F displacement = 0.f;
		if (!b1 || !b2){
			VEC3F displacement = p1 + (i * spacing_a) * unitdiff_a;
			InitParticle(displacement, normal, unitdiff_a, onlyCountParticles, false, 0, false, isInner);
		}
		for (int j = 1; j < lineCnt_b; j++){
			VEC3F disp2 = p1 + (i * spacing_a) * unitdiff_a + (j * spacing_b) * unitdiff_b;
			InitParticle(disp2, normal, unitdiff_a, onlyCountParticles, false, 0, false, isInner);
		}
		if (!b4 || !b3){
			VEC3F displacement = p4 + (i * spacing_a) * unitdiff_a;
			InitParticle(displacement, normal, unitdiff_a, onlyCountParticles, false, 0, false, isInner);
		}
	}

	if (b2 && b3){
		if (onlyCountParticles)
		{
			overlappingLine ol = { 0, 0, p2, p3, -normal, normal.cross(unitdiff_b) };
			sph->insertOverlappingLines(ol);
		}
	}
	else{
		for (int i = 1; i < lineCnt_b; i++){
			VEC3F displacement = p2 + (i * spacing_b) * unitdiff_b;
			InitParticle(displacement, normal, unitdiff_a, onlyCountParticles, false, 0, false, isInner);
		}
	}

	if (b1 && b2){
		if (onlyCountParticles)
		{
			overlappingLine ol = { 0, 0, p1, p2, -normal, -normal.cross(unitdiff_a) };
			sph->insertOverlappingLines(ol);
		}
	}
// 
	if (b4 && b3){
		if (onlyCountParticles){
			overlappingLine ol = { 0, 0, p3, p4, -normal, normal.cross(unitdiff_a) };
			sph->insertOverlappingLines(ol);
		}
	}
// 
// 	
// 
// 	if (b2 || b3){
// 		if (onlyCountParticles)
// 		{
// 			overlappingLine ol = { 0, 0, p2, p3, -normal, -normal.cross(unitdiff_b) };
// 			sph->insertOverlappingLines(ol);
// 		}
// 	}

}

bool plane::particleCollision(const VEC3F& position, float radius)
{
	return true;
	//return utils::circleLineIntersect(startPoint, endPoint, position, radius);
}

std::vector<corner> plane::corners()
{
	corner c1 = { p1, normal, (p1 - p2).normalize() };
	corner c2 = { p2, normal, (p2 - p3).normalize() };
	corner c3 = { p3, normal, (p3 - p4).normalize() };
	corner c4 = { p4, normal, (p4 - p1).normalize() };
	std::vector<corner> list(4);
	list[0] = c1;
	list[1] = c2;
	list[2] = c3;
	list[3] = c4;
	return list;
// 	corner c1 = { startPoint, normal, (startPoint - endPoint).normalize() };
// 	corner c2 = { endPoint, normal, (endPoint - startPoint).normalize() };
// 	std::vector<corner> list(2);
// 	list[0] = c1;
// 	list[1] = c2;
// 	return list;
}

void plane::initExpression()
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