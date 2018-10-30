#include "vobject.h"

int vobject::count = -1;

vobject::vobject()
	: outPos(NULL)
	, outRot(NULL)
	, select_cube(NULL)
	, vot(VIEW_OBJECT)
	, drawingMode(GL_LINE)
	, type(V_OBJECT)
	, m_type(NO_MATERIAL)
	, display(false)
	, isSelected(false)
	, ixx(0), iyy(0), izz(0)
	, ixy(0), ixz(0), iyz(0)
	, mass(0)
	, vol(0)
{
	count++;
	id = count;
	clr = colors[count];
}

vobject::vobject(Type tp, QString _name)
	: nm(_name)
	, outPos(NULL)
	, outRot(NULL)
	, select_cube(NULL)
	, vot(VIEW_OBJECT)
	, drawingMode(GL_LINE)
	, type(tp)
	, m_type(NO_MATERIAL)
	, display(false)
	, isSelected(false)
	, ixx(0), iyy(0), izz(0)
	, ixy(0), ixz(0), iyz(0)
	, mass(0)
	, vol(0)
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

void vobject::setColor(color_type ct)
{
	clr = colors[(int)ct];
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

void vobject::updateView(VEC3D& _pos, VEC3D& _ang)
{
	pos0 = _pos;
	ang0 = _ang;
	if (select_cube)
		select_cube->updateView(_pos, _ang);
}

void vobject::animationFrame(VEC3D& p, EPD& ep)
{
	//unsigned int f = vcontroller::getFrame();
	glTranslated(p.x, p.y, p.z);
	VEC3D e = ep2e(ep);
	double xi = (e.x * 180) / M_PI;
	double th = (e.y * 180) / M_PI;
	double ap = (e.z * 180) / M_PI;
	double diff = xi + ap;
	glRotated(xi, 0, 0, 1);
	glRotated(th, 1, 0, 0);
	glRotated(ap, 0, 0, 1);
}