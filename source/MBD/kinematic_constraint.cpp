#include "kinematic_constraint.h"

using namespace parSIM;

kinematicConstraint::kinematicConstraint()
	: i(0)
	, j(0)
	, lm(NULL)
	, reactionForce(NULL)
	, sRow(0)
	, iCol(0)
	, jCol(0)
	, nConst(0)
	, max_nnz(0)
	, principal_axis(0)
{
	loc = 0.0;
	axis = 0.0;

	sp_i = 0.0;
	sp_j = 0.0;
	f_i = 0.0;
	f_j = 0.0;
	h_i = 0.0; 
	h_j = 0.0;
	g_i = 0.0; 
	g_j = 0.0;
};

kinematicConstraint::kinematicConstraint(const std::string& _name, int id, tjoint jType, vector3d location, 
										 int ip, vector3d fi, vector3d gi,
										 int jp, vector3d fj, vector3d gj)
	: i(0)
	, j(0)
	, lm(NULL)
	, reactionForce(NULL)
	, sRow(0)
	, iCol(0)
	, jCol(0)
	, nConst(0)
	, max_nnz(0)
	, principal_axis(0)
{
	type = jType;
	name = _name;
	//iName = ip;
	//jName = jp;
	i = ip;
	j = jp;
	ID = id;
	axis = 0.0;

	sp_i = 0.0;
	sp_j = 0.0;
	f_i = fi;
	f_j = fj;
	g_i = gi;
	g_j = gj;
	h_i = fi.cross(gi);
	h_j = fj.cross(gj);
	loc = location;

	//setCoordinates();
}

kinematicConstraint::kinematicConstraint(const kinematicConstraint& kc)
	: i(kc.i)
	, j(kc.j)
	, lm(kc.lm)
	, reactionForce(NULL)
	, principal_axis(kc.principal_axis)
{
	axis = kc.axis;
	principal_axis = kc.principal_axis;
	name = kc.name;
	loc = kc.loc;
	type = kc.type;
	sp_i = kc.sp_i;
	sp_j = kc.sp_j;
	f_i = kc.f_i;
	f_j = kc.f_j;
	h_i = kc.h_i;
	h_j = kc.h_j;
	g_i = kc.g_i;
	g_j = kc.g_j;
	iName = kc.iName;
	jName = kc.jName;
	sRow = kc.sRow;
	iCol = kc.iCol;
	jCol = kc.jCol;
	nConst = kc.nConst;
	max_nnz = kc.max_nnz;
}

kinematicConstraint::~kinematicConstraint()
{
	if(lm) lm = NULL;
	if(reactionForce) delete [] reactionForce; reactionForce = NULL;
}

void kinematicConstraint::setCoordinates(pointmass& ip, pointmass& jp)
{
	sp_i = loc - ip.Position();
	sp_j = loc - jp.Position();
	sp_i = ip.toLocal(sp_i);
	sp_j = jp.toLocal(sp_j);
	switch(type){
	case REVOLUTE: 
		nConst = 5;
		(i && j) ? max_nnz += 46 : max_nnz += 23; 
		break;
	case SPHERICAL: 
		nConst = 3;
		(i && j) ? max_nnz += 30 : max_nnz += 15; 
		break;
	case CYLINDERICAL: 
		nConst = 4;
		(i && j) ? max_nnz += 44 : max_nnz += 22;
		break;
	case TRANSLATION: 
		nConst = 5;
		(i && j) ? max_nnz += 52 : max_nnz += 26;
		break;
	}
}

void kinematicConstraint::allocResultMemory(int ndata)
{
	reactionForce = new double[ndata*nConst];
}

// QDataStream &operator<<(QDataStream &out, const kinematicConstraint &kc)
// {
// // 	out << kc.name
// // 		<< kc.i->name
// // 		<< kc.j->name
// // 		<< quint32((int)kc.jt.type)
// // 		<< quint32(kc.principal_axis)
// // 		<< kc.axis.x << kc.axis.y << kc.axis.z
// // 		<< kc.loc.x << kc.loc.y << kc.loc.z;
//  	return out;
// }
// 
// QDataStream &operator>>(QDataStream &in, kinematicConstraint &kc)
// {
// // 	quint32 intType;
// // 	quint32 paxis;
// // 	in  >> kc.name
// // 		>> kc.iName
// // 		>> kc.jName
// // 		>> intType
// // 		>> paxis
// // 		>> kc.axis.x >> kc.axis.y >> kc.axis.z
// // 		>> kc.loc.x >> kc.loc.y >> kc.loc.z;
// // 	kc.jt.type = (kinematicConstraint::tjoint)intType;
// // 	kc.principal_axis = paxis;
// // 	//kc.setCoordinates();
//  	return in;
// }
