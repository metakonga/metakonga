#include "mass.h"

mass::mass()
	: ms(0.0)
	, id(0)
	, md(NULL)
	, vobj(NULL)
	, vpobj(NULL)
{
	ep.e0 = 1.0;
	
}

mass::mass(modeler* _md, QString& _name)
	: ms(0.0)
	, id(0)
	, md(_md)
	, nm(_name)
	, vobj(NULL)
	, vpobj(NULL)
{
	ep.e0 = 1.0;
	makeTransformationMatrix();
	//setInertia();
}

mass::~mass()
{
}

void mass::makeTransformationMatrix()
{
	A.a00 = 2 * (ep.e0*ep.e0 + ep.e1*ep.e1 - 0.5);	A.a01 = 2 * (ep.e1*ep.e2 - ep.e0*ep.e3);		A.a02 = 2 * (ep.e1*ep.e3 + ep.e0*ep.e2);
	A.a10 = 2 * (ep.e1*ep.e2 + ep.e0*ep.e3);		A.a11 = 2 * (ep.e0*ep.e0 + ep.e2*ep.e2 - 0.5);	A.a12 = 2 * (ep.e2*ep.e3 - ep.e0*ep.e1);
	A.a20 = 2 * (ep.e1*ep.e3 - ep.e0*ep.e2);		A.a21 = 2 * (ep.e2*ep.e3 + ep.e0*ep.e1);		A.a22 = 2 * (ep.e0*ep.e0 + ep.e3*ep.e3 - 0.5);
}

void mass::saveData(QTextStream& ts) const
{
	ts << ms << " " << prin_iner.x << " " << prin_iner.y << " " << prin_iner.z << endl
		<< sym_iner.x << " " << sym_iner.y << " " << sym_iner.z << endl
		<< pos.x << " " << pos.y << " " << pos.z << endl
		<< vel.x << " " << vel.y << " " << vel.z << endl
		<< ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3 << endl;
}

void mass::openData(QTextStream& ts)
{
	ts >> ms >> prin_iner.x >> prin_iner.y >> prin_iner.z
		>> sym_iner.x >> sym_iner.y >> sym_iner.z;
		//>> pos.x >> pos.y >> pos.z
		//>> vel.x >> vel.y >> vel.z
		//>> ep.e0 >> ep.e1 >> ep.e2 >> ep.e3;
	makeTransformationMatrix();
	setInertia();
}

VEC3D mass::toLocal(VEC3D &v)
{
	VEC3D tv;
	tv.x = A.a00*v.x + A.a10*v.y + A.a20*v.z;
	tv.y = A.a01*v.x + A.a11*v.y + A.a21*v.z;
	tv.z = A.a02*v.x + A.a12*v.y + A.a22*v.z;
	return tv;
}

VEC3D mass::toGlobal(VEC3D &v)
{
	VEC3D tv;
	tv.x = A.a00*v.x + A.a01*v.y + A.a02*v.z;
	tv.y = A.a10*v.x + A.a11*v.y + A.a12*v.z;
	tv.z = A.a20*v.x + A.a21*v.y + A.a22*v.z;
	return tv;
}
