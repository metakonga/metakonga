#include "vobject.h"

int vobject::count = -1;

vobject::vobject()
	: outPos(NULL)
	, outRot(NULL)
{
	count++;
	id = count;
	clr = colors[count];
}

vobject::vobject(QString& _name)
	: nm(_name)
	, outPos(NULL)
	, outRot(NULL)
{
	count++;
	id = count;
	clr = colors[count];
	clr.setAlphaF(0.5f);
}

vobject::~vobject()
{
	count--;
	if (outPos) delete[] outPos; outPos = NULL;
	if (outRot) delete[] outRot; outRot = NULL;
}

void vobject::msgBox(QString ch, QMessageBox::Icon ic)
{
	QMessageBox msg;
	msg.setIcon(ic);
	msg.setText(ch);
	msg.exec();
}

void vobject::setResultData(unsigned int n)
{
	if (!outPos)
		outPos = new VEC3D[n];
	if (!outRot)
		outRot = new EPD[n];
}

void vobject::insertResultData(unsigned int i, VEC3D& p, EPD& r)
{
	outPos[i] = p;
	outRot[i] = r;
}

void vobject::copyCoordinate(GLuint _coord)
{
	coord = _coord;
}