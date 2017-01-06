#include "massCalculator.h"

using namespace parSIM;

massCalculator::massCalculator()
{

}

massCalculator::~massCalculator()
{

}

void massCalculator::calculateMassMatrix(matrix<double>& lhs, std::map<std::string, pointmass*>& masses, double mul)
{
 	int cnt=0;
 	lhs.zeros();
 	for(std::map<std::string, pointmass*>::iterator it = masses.begin(); it != masses.end(); it++){
		if(!it->second->ID()) continue;
		lhs(cnt,cnt) = lhs(cnt+1,cnt+1) = lhs(cnt+2,cnt+2) = mul*it->second->Mass();
		globalizedInertia(lhs, it->second->Inertia(), it->second->Orientation(), cnt, mul);
		cnt += 7;
	}
	//lhs.display();
}

void massCalculator::globalizedInertia(matrix<double>& out, matrix3x3<double>& J, euler_parameter<double>& ep, int i, double mul)
{
	i += 3;
	double mul4 = mul*4;
	matrix4x4<double> LTJL = mul4*transpose(G(ep), J*G(ep));
	for(int j(0); j < 4; j++){
		for(int k(0); k < 4; k++){
			if(LTJL(j,k)) out(i+j,i+k) = LTJL(j,k);
		}
	}
// 	out(i,i)   = mul4*LTJL.a00; out(i,i+1)   = mul4*LTJL.a01; out(i,i+2)   = mul4*LTJL.a02; out(i,i+3)   = mul4*LTJL.a03;
// 	out(i+1,i) = mul4*LTJL.a10; out(i+1,i+1) = mul4*LTJL.a11; out(i+1,i+2) = mul4*LTJL.a12; out(i+1,i+3) = mul4*LTJL.a13;
// 	out(i+2,i) = mul4*LTJL.a20; out(i+2,i+1) = mul4*LTJL.a21; out(i+2,i+2) = mul4*LTJL.a22; out(i+2,i+3) = mul4*LTJL.a23;
// 	out(i+3,i) = mul4*LTJL.a30; out(i+3,i+1) = mul4*LTJL.a31; out(i+3,i+2) = mul4*LTJL.a32; out(i+3,i+3) = mul4*LTJL.a33;
}

void massCalculator::calculateSystemJacobian(matrix<double>& lhs, std::map<std::string, pointmass*> &masses, double mul)
{
 	//int cnt = masses.size() * 7;
 	int sRow = 0;
 	euler_parameter<double> e;
 	euler_parameter<double> edd;
 	matrix3x3<double> inertia;
 	matrix4x4<double> data;
	unsigned int i = 0;
	for(std::map<std::string, pointmass*>::iterator mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		sRow = i * 7 + 3;
		e = mass->Orientation();
		edd = mass->ddOrientation();
		inertia = mass->Inertia();
		data = mul*(-transpose(G(e), inertia*G(edd)) + opMiner(inertia*(G(e)*edd))); 
		lhs.plus(sRow, sRow, POINTER(data), MAT4x4);
		i++;
	}
}