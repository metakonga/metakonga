#include "kinematicConstraint.h"
#include "mbd_model.h"


kinematicConstraint::kinematicConstraint()
	: ib(NULL)
	, jb(NULL)
	, lm(NULL)
	//, reactionForce(NULL)
	, srow(0)
	, icol(0)
	, jcol(0)
	, nconst(0)
	, maxnnz(0)
	//, principal_axis(0)
{

}

kinematicConstraint::kinematicConstraint(
	mbd_model *_md, QString& _nm, kinematicConstraint::Type kt,
	pointMass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi, 
	pointMass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj)
	: md(_md)
	, ib(ip)
	, jb(jp)
	, lm(NULL)
	, type(kt)
	, nm(_nm)
	//, reactionForce(NULL)
	, srow(0)
	, icol(0)
	, jcol(0)
	, nconst(0)
	, maxnnz(0)
	, fi(_fi)
	, fj(_fj)
	, gi(_gi)
	, gj(_gj)
	, spi(_spi)
	, spj(_spj)
										 //, principal_axis(0)
{
	hi = fi.cross(gi);
	hj = fj.cross(gj);
	setCoordinates();
}

kinematicConstraint::kinematicConstraint(const kinematicConstraint& _kc)
{
	ib = _kc.iMass();
	jb = _kc.jMass();
	ax = _kc.axis();
	nm = _kc.name();
	type = _kc.constType();
	spi = _kc.sp_i();
	spj = _kc.sp_j();
	hi = _kc.h_i();
	hj = _kc.h_j();
	gi = _kc.g_i();
	gj = _kc.g_j();
	fi = _kc.f_i();
	fj = _kc.f_j();
	srow = _kc.startRow();
	icol = _kc.iColumn();
	jcol = _kc.jColumn();
	nconst = _kc.numConst();
	maxnnz = _kc.maxNNZ();
}

kinematicConstraint::kinematicConstraint(QString _nm)
	: nm(_nm)
	, ib(NULL)
	, jb(NULL)
	, lm(NULL)
	//, reactionForce(NULL)
	, srow(0)
	, icol(0)
	, jcol(0)
	, nconst(0)
	, maxnnz(0)
{

}

kinematicConstraint::~kinematicConstraint()
{

}

void kinematicConstraint::setLocation(VEC3D& _loc)
{
	loc = _loc;
}

VEC3D kinematicConstraint::location() const
{
	return loc;
}

VEC3D kinematicConstraint::CurrentDistance()
{
	return jb->Position() + jb->toGlobal(spj) - ib->Position() - ib->toGlobal(spi);
}

void kinematicConstraint::setCoordinates()
{
// 	spi = loc - (i ? i->Position() : VEC3D(0, 0, 0));
// 	spj = loc - (j ? j->Position() : VEC3D(0, 0, 0));
//  	spi = i->toLocal(spi - i->Position());
//  	spj = j->toLocal(spj - j->Position());
	bool incGround = false;
	if (ib->MassType() == pointMass::GROUND || jb->MassType() == pointMass::GROUND)
	{
		incGround = true;
	}
	switch (type){
	case kinematicConstraint::FIXED:
		nconst = 6;
		!incGround ? maxnnz += 54 : maxnnz += 27;
		break;
	case kinematicConstraint::REVOLUTE:
		nconst = 5;
		!incGround ? maxnnz += 46 : maxnnz += 23;
		break;
	case kinematicConstraint::TRANSLATIONAL:
		nconst = 5;
		!incGround ? maxnnz += 52 : maxnnz += 26;
		break;
	case kinematicConstraint::SPHERICAL:
		nconst = 3;
		!incGround ? maxnnz += 30 : maxnnz += 15;
		break; 
	case kinematicConstraint::UNIVERSAL:
		nconst = 4;
		!incGround ? maxnnz += 38 : maxnnz += 19;
		break;
	}
}

// void kinematicConstraint::calcReactionForceResult(double ct)
// {
// 
// }

void kinematicConstraint::setLagrangeMultiplierPointer(double* r)
{
	lm = r;
}

void kinematicConstraint::saveData(QTextStream& qts)
{
	qts << "ELEMENT " << "constraint" << endl
		<< "NAME " << nm << endl
		<< "TYPE " << type << endl
		<< "FIRST_BODY " << ib->Name() << endl
		<< "SECOND_BODY " << jb->Name() << endl
		<< "FIRST_JOINT_COORDINATE "
		<< spi.x << " " << spi.y << " " << spi.z << " "
		<< fi.x << " " << fi.y << " " << fi.z << " "
		<< gi.x << " " << gi.y << " " << gi.z << endl
		<< "SECOND_JOINT_COORDINATE "
		<< spj.x << " " << spj.y << " " << spj.z << " "
		<< fj.x << " " << fj.y << " " << fj.z << " "
		<< gj.x << " " << gj.y << " " << gj.z << endl;
// 	qts << endl << "KINEMATIC_CONSTRAINT" << endl		
// 		<< "NAME " << nm << endl
// 		<< "TYPE " << (int)type << endl
// 		<< "LOCATION " << loc.x << " " << loc.y << " " << loc.z << endl
// 		<< "BASE_NAME " << ib->Name() << endl		
// 		<< "BASE_LOCAL " << spi.x << " " << spi.y << " " << spi.z << endl
// 		<< "BASE_P_POINT " << fi.x << " " << fi.y << " " << fi.z << endl
// 		<< "BASE_Q_POINT " << gi.x << " " << gi.y << " " << gi.z << endl
// 		<< "ACTION_NAME " << jb->Name() << endl
// 		<< "ACTION_LOCAL " << spj.x << " " << spj.y << " " << spj.z << endl
// 		<< "ACTION_P_POINT " << fj.x << " " << fj.y << " " << fj.z << endl
// 		<< "ACTION_Q_POINT " << gj.x << " " << gj.y << " " << gj.z << endl;
}

void kinematicConstraint::exportResultData2TXT()
{
// 	QString file_name = model::path + model::name + "/" + nm + ".txt";
// 	QFile qf(file_name);
// 	qf.open(QIODevice::WriteOnly);
// 	QTextStream qts(&qf);
// 	qts << "time "
// 		<< "fix " << "fiy " << "fiz " << "ri0 " << "ri1 " << "ri2 " << "ri3 "
// 		<< "fjx " << "fjy " << "fjz " << "rj0 " << "rj1 " << "rj2 " << "rj3 " << endl;
// 	foreach(reactionForceData p, model::rs->reactionForceResults()[nm])
// 	{
// 		qts << p.time
// 			<< " " << p.iAForce.x << " " << p.iAForce.y << " " << p.iAForce.z
// 			<< " " << p.iRForce.x << " " << p.iRForce.y << " " << p.iRForce.z << " " << p.iRForce.w
// 			<< " " << p.jAForce.x << " " << p.jAForce.y << " " << p.jAForce.z
// 			<< " " << p.jRForce.x << " " << p.jRForce.y << " " << p.jRForce.z << " " << p.jRForce.w << endl;
// 	}
// 	qf.close();
}

void kinematicConstraint::setZeroLagrangeMultiplier()
{
	memset(lm, 0, sizeof(double) * nconst);
}
