#include "pointMass.h"
#include "model.h"
#include <QFile>
#include <QTextStream>
#include <QString>

pointMass::pointMass()
	: object()
	, ms(0)
	, ang(0.0)
	, type(RIGID_BODY)
{
	ep.e0 = 1.0;
	count++;
}

pointMass::pointMass(QString& _name, geometry_type gt, geometry_use gu)
	: object(_name, gt, gu)
	, ang(0.0)
	, ms(0)
	, type(RIGID_BODY)
{
// 	if (gt == POLYGON_SHAPE)
// 		type = POLYMER;
	ep.e0 = 1.0;
	makeTransformationMatrix();
	count++;
}

pointMass::pointMass(QString& _name)
	: object(_name, NO_GEOMETRY_TYPE, MASS)
	, ang(0.0)
	, ms(0)
	, type(RIGID_BODY)
{
	ep.e0 = 1.0;
	makeTransformationMatrix();
	count++;
	//setInertia();
}

pointMass::~pointMass()
{
}

void pointMass::makeTransformationMatrix()
{
	A.a00 = 2 * (ep.e0*ep.e0 + ep.e1*ep.e1 - 0.5);	A.a01 = 2 * (ep.e1*ep.e2 - ep.e0*ep.e3);		A.a02 = 2 * (ep.e1*ep.e3 + ep.e0*ep.e2);
	A.a10 = 2 * (ep.e1*ep.e2 + ep.e0*ep.e3);		A.a11 = 2 * (ep.e0*ep.e0 + ep.e2*ep.e2 - 0.5);	A.a12 = 2 * (ep.e2*ep.e3 - ep.e0*ep.e1);
	A.a20 = 2 * (ep.e1*ep.e3 - ep.e0*ep.e2);		A.a21 = 2 * (ep.e2*ep.e3 + ep.e0*ep.e1);		A.a22 = 2 * (ep.e0*ep.e0 + ep.e3*ep.e3 - 0.5);
}

void pointMass::makeTransformationMatrix2D()
{
	A.a00 = cos(pos.z); A.a01 = -sin(pos.z);
	A.a10 = sin(pos.z); A.a11 = cos(pos.z);
}

MAT33D pointMass::getTransformationMatrix()
{
	return A;
}

void pointMass::exportResultData2TXT()
{
// 	QString file_name = model::path + model::name + "/" + nm + ".txt";
// 	QFile qf(file_name);
// 	qf.open(QIODevice::WriteOnly);
// 	QTextStream qts(&qf);
// 	qts << "time "
// 		<< "px " << "py " << "pz " << "ep0 " << "ep1 " << "ep2 " << "ep3 "
// 		<< "vx " << "vy " << "vz " << "wx " << "wy " << "wz "
// 		<< "ax " << "ay " << "az " << "apx " << "apy " << "apz " << endl;
// 	foreach(pointMassResultData p, pmrd)
// 	{
// 		qts << p.time
// 			<< " " << p.pos.x << " " << p.pos.y << " " << p.pos.z
// 			<< " " << p.ep.e0 << " " << p.ep.e1 << " " << p.ep.e2 << " " << p.ep.e3
// 			<< " " << p.vel.x << " " << p.vel.y << " " << p.vel.z
// 			<< " " << p.omega.x << " " << p.omega.y << " " << p.omega.z
// 			<< " " << p.acc.x << " " << p.acc.y << " " << p.acc.z
// 			<< " " << p.alpha.x << " " << p.alpha.y << " " << p.alpha.z << endl;
// 	}
// 	qf.close();
}

VEC3D pointMass::getAngularVelocity()
{
	return 2.0 * ep.G() * ev;
}

void pointMass::setSymetryInertia(double ixy, double ixz, double iyz)
{
	sym_iner = VEC3D(ixy, ixz, iyz);
}

void pointMass::setDiagonalInertia(double ixx, double iyy, double izz)
{
	diag_iner = VEC3D(ixx, iyy, izz);
}

void pointMass::setRotationalVelocity(VEC3D& rv)
{
	VEC4D _ev = 0.5 * transpose(ep.G(), rv);
	ev = EPD(_ev.x, _ev.y, _ev.z, _ev.w);
}

void pointMass::saveData(QTextStream& qts)
{
	qts << "ELEMENT " << "point_mass" << endl
		<< "NAME " << name << endl
		<< "MASS " << ms << endl
		<< "MATERIAL_TYPE " << mat_type << endl
		<< "POSITION " << pos.x << " " << pos.y << " " << pos.z << endl
		<< "PARAMETER " << ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3 << endl
		<< "D_INERTIA " << diag_iner.x << " " << diag_iner.y << " " << diag_iner.z << endl
		<< "S_INERTIA " << sym_iner.x << " " << sym_iner.y << " " << sym_iner.z << endl;
// 	qts << endl;// << "TYPE ";
// 	switch (type)
// 	{
// 	case RIGID:	qts << "RIGID" << endl; break;
// 	case FLEXIBLE: qts << "FLEXIBLE" << endl; break;
// 	}
// 	qts << "NAME " << nm << endl;
// 	qts << "MASS " << ms << endl
// 		<< "PINER " << prin_iner.x << " " << prin_iner.y << " " << prin_iner.z << endl
// 		<< "SINER " << sym_iner.x << " " << sym_iner.y << " " << sym_iner.z << endl
// 		<< "POSITION " << pos.x << " " << pos.y << " " << pos.z << endl
// 		<< "VELOCITY " << vel.x << " " << vel.y << " " << vel.z << endl
// 		<< "EPARAMETER " << ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3 << endl;
}

void pointMass::openData(QTextStream& qts)
{
// 	QString ch;
// 	qts >> ch >> ms
// 		>> ch >> prin_iner.x >> prin_iner.y >> prin_iner.z
// 		>> ch >> sym_iner.x >> sym_iner.y >> sym_iner.z
// 		>> ch >> pos.x >> pos.y >> pos.z
// 		>> ch >> vel.x >> vel.y >> vel.z
// 		>> ch >> ep.e0 >> ep.e1 >> ep.e2 >> ep.e3;
// 		//>> pos.x >> pos.y >> pos.z
// 		//>> vel.x >> vel.y >> vel.z
// 		//>> ep.e0 >> ep.e1 >> ep.e2 >> ep.e3;
// 	makeTransformationMatrix();
// 	setInertia();
}

VEC3D pointMass::toLocal(VEC3D &v)
{
	VEC3D tv;
	tv.x = A.a00*v.x + A.a10*v.y + A.a20*v.z;
	tv.y = A.a01*v.x + A.a11*v.y + A.a21*v.z;
	tv.z = A.a02*v.x + A.a12*v.y + A.a22*v.z;
	return tv;
}

VEC3D pointMass::toLocal2D(VEC3D &v)
{
	VEC3D tv;
	tv.x = A.a00*v.x + A.a10*v.y;
	tv.y = A.a01*v.x + A.a11*v.y;
	tv.z = 0;
	return tv;
}

VEC3D pointMass::toGlobal2D(VEC3D &v)
{
	VEC3D tv;
	tv.x = A.a00*v.x + A.a01*v.y;
	tv.y = A.a10*v.x + A.a11*v.y;
	tv.z = 0;
	return tv;
}

VEC3D pointMass::toGlobal(VEC3D &v)
{
	VEC3D tv;
	tv.x = A.a00*v.x + A.a01*v.y + A.a02*v.z;
	tv.y = A.a10*v.x + A.a11*v.y + A.a12*v.z;
	tv.z = A.a20*v.x + A.a21*v.y + A.a22*v.z;
	return tv;
}
// 
// VEC3D pointMass::toGlobal(VEC3F &v)
// {
// 	VEC3D tv;
// 	tv.x = A.a00*v.x + A.a01*v.y + A.a02*v.z;
// 	tv.y = A.a10*v.x + A.a11*v.y + A.a12*v.z;
// 	tv.z = A.a20*v.x + A.a21*v.y + A.a22*v.z;
// 	return tv;
// }

VEC3D pointMass::toDGlobal(VEC3D &v)
{
	MAT34D dL = ev.L();
	MAT34D G = ep.G();
	MAT33D dA = MAT33D(
		dL.a00 * G.a00 + dL.a01 * G.a01 + dL.a02 * G.a02 + dL.a03 * G.a03, dL.a00 * G.a10 + dL.a01 * G.a11 + dL.a02 * G.a12 + dL.a03 * G.a13, dL.a00 * G.a20 + dL.a01 * G.a21 + dL.a02 * G.a22 + dL.a03 * G.a23,
		dL.a10 * G.a00 + dL.a11 * G.a01 + dL.a12 * G.a02 + dL.a13 * G.a03, dL.a10 * G.a10 + dL.a11 * G.a11 + dL.a12 * G.a12 + dL.a13 * G.a13, dL.a10 * G.a20 + dL.a11 * G.a21 + dL.a12 * G.a22 + dL.a13 * G.a23,
		dL.a20 * G.a00 + dL.a21 * G.a01 + dL.a22 * G.a02 + dL.a23 * G.a03, dL.a20 * G.a10 + dL.a21 * G.a11 + dL.a22 * G.a12 + dL.a23 * G.a13, dL.a20 * G.a20 + dL.a21 * G.a21 + dL.a22 * G.a22 + dL.a23 * G.a23);
	return VEC3D(
		dA.a00 * v.x + dA.a01 * v.y + dA.a02 * v.z,
		dA.a10 * v.x + dA.a11 * v.y + dA.a12 * v.z,
		dA.a20 * v.x + dA.a21 * v.y + dA.a22 * v.z);
}

double pointMass::rotationAngle()
{
	return ang;
}

// void pointMass::setDataFromFile(QTextStream& qts)
// {
// 	
// }
