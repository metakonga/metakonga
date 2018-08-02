#include "vobject.h"

int vobject::count = -1;

vobject::vobject()
	: outPos(NULL)
	, outRot(NULL)
	, drawingMode(GL_FILL)
	, isSelected(false)
{
	count++;
	id = count;
	clr = colors[count];
}

vobject::vobject(QString& _name)
	: nm(_name)
	, outPos(NULL)
	, outRot(NULL)
	, drawingMode(GL_FILL)
	, isSelected(false)
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

void vobject::animationFrame(float ox, float oy, float oz)
{
	unsigned int f = vcontroller::getFrame();
	glTranslated(outPos[f].x + ox, outPos[f].y + oy, outPos[f].z + oz);
	VEC3D e = ep2e(outRot[f]);
	double xi = (e.x * 180) / M_PI;
	double th = (e.y * 180) / M_PI;
	double ap = (e.z * 180) / M_PI;
	double diff = xi + ap;
	glRotated(xi, 0, 0, 1);
	glRotated(th, 1, 0, 0);
	glRotated(ap, 0, 0, 1);
}