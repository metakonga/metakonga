#include "vplane.h"
#include <QTextStream>

vplane::vplane()
	: vobject()
{

}

vplane::vplane(QString& _name)
	: vobject(V_PLANE, _name)
{

}

vplane::vplane(QTextStream& in)
	: vobject()
{
	QString ch;
	in >> ch; nm = ch;//this->setObjectName(ch);
	//in >> idr >> idm;

	makePlaneGeometry(in);
}

bool vplane::makePlaneGeometry(QTextStream& in)
{
	float l1, l2;
	float u1[3];
	float u2[3];
	float uw[3];
	float xw[3];
	float pa[3];
	float pb[3];
	float pc[3];

	in >> l1 >> l2
		>> u1[0] >> u1[1] >> u1[2]
		>> u2[0] >> u2[1] >> u2[2]
		>> uw[0] >> uw[1] >> uw[2]
		>> xw[0] >> xw[1] >> xw[2]
		>> pa[0] >> pa[1] >> pa[2]
		>> pb[0] >> pb[1] >> pb[2];
	p0[0] = xw[0]; p0[1] = xw[1]; p0[2] = xw[2];
	p1[0] = pa[0] + p0[0]; p1[1] = pa[1] + p0[1]; p1[2] = pa[2] + p0[2];
	p3[0] = pb[0] + p0[0]; p3[1] = pb[1] + p0[1]; p3[2] = pb[2] + p0[2];
	p2[0] = p0[0] + pb[0] + l2 * u1[0];
	p2[1] = p0[1] + pb[1] + l2 * u1[1];
	p2[2] = p0[2] + pb[2] + l2 * u1[2];


	display = define();
	return true;
}

bool vplane::makePlaneGeometry(VEC3D& pa, VEC3D& pb, VEC3D& pc, VEC3D& pd)
{
	float l1 = 0.0f;
	float l2 = 0.0f;
	l1 = (float)(pb - pa).length();
	l2 = (float)(pc - pb).length();
// 	p0[0] = -0.5f * l1; p0[1] = 0.0f; p0[2] = -0.5f * l2;
// 	p1[0] = -0.5f * l1; p1[1] = 0.0f; p1[2] =  0.5f * l2;
// 	p2[0] =  0.5f * l1; p2[1] = 0.0f; p2[2] =  0.5f * l2;
// 	p3[0] =  0.5f * l1; p3[1] = 0.0f; p3[2] = -0.5f * l2;
// 	pos0 = 
	pos0 = 0.5 * (pc + pa);
	MAT33D eang = eulerAngle(0, 0, 0);
	VEC3D lpa = global2local_eulerAngle(eang, (pa - pos0));
	VEC3D lpb = global2local_eulerAngle(eang, (pb - pos0));
	VEC3D lpc = global2local_eulerAngle(eang, (pc - pos0));
	VEC3D lpd = global2local_eulerAngle(eang, (pd - pos0));
	p0[0] = (float)lpa.x; p0[1] = (float)lpa.y; p0[2] = (float)lpa.z;
	p1[0] = (float)lpb.x; p1[1] = (float)lpb.y; p1[2] = (float)lpb.z;
	p2[0] = (float)lpc.x; p2[1] = (float)lpc.y; p2[2] = (float)lpc.z;
	p3[0] = (float)lpd.x; p3[1] = (float)lpd.y; p3[2] = (float)lpd.z;
	ixx += lpa.y * lpa.y + lpa.z * lpa.z;
	iyy += lpa.x * lpa.x + lpa.z * lpa.z;
	izz += lpa.x * lpa.x + lpa.y * lpa.y;

	ixx += lpb.y * lpb.y + lpb.z * lpb.z;
	iyy += lpb.x * lpb.x + lpb.z * lpb.z;
	izz += lpb.x * lpb.x + lpb.y * lpb.y;

	ixx += lpc.y * lpc.y + lpc.z * lpc.z;
	iyy += lpc.x * lpc.x + lpc.z * lpc.z;
	izz += lpc.x * lpc.x + lpc.y * lpc.y;

	ixx += lpd.y * lpd.y + lpd.z * lpd.z;
	iyy += lpd.x * lpd.x + lpd.z * lpd.z;
	izz += lpd.x * lpd.x + lpd.y * lpd.y;
	ixx /= 12.0;
	iyy /= 12.0;
	izz /= 12.0;
	//ixx = (1.0 / 12.0) * ()
	vol = l1 * l2;
// 	p0[0] = xw.x; p0[1] = xw.y; p0[2] = xw.z;
// 	p1[0] = pa.x + p0[0]; p1[1] = pa.y + p0[1]; p1[2] = pa.z + p0[2];
// 	p3[0] = pb.x + p0[0]; p3[1] = pb.y + p0[1]; p3[2] = pb.z + p0[2];
// 	p2[0] = p0[0] + pb.x + l * u1.x;
// 	p2[1] = p0[1] + pb.y + l * u1.y;
// 	p2[2] = p0[2] + pb.z + l * u1.z;
	display = this->define();
	return true;
}

void vplane::draw(GLenum eMode)
{
	//qDebug() << nm << " is displayed - " << glList << " - " << display;
	if (display){
		//qDebug() << nm << " is displayed - " << glList;
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glPushMatrix();
		glDisable(GL_LIGHTING);
		glColor3f(clr.redF(), clr.greenF(), clr.blueF());
		if (eMode == GL_SELECT)
			glLoadName((GLuint)ID());
		glTranslated(pos0.x, pos0.y, pos0.z);
		glRotated(ang0.x, 0, 0, 1);
		glRotated(ang0.y, 1, 0, 0);
		glRotated(ang0.z, 0, 0, 1);
		glCallList(glList);
		if (isSelected)
		{
			glLineWidth(2.0);
			glLineStipple(5, 0x5555);
			glEnable(GL_LINE_STIPPLE);
			glColor3f(1.0f, 0.0f, 0.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glCallList(glHiList);
			glDisable(GL_LINE_STIPPLE);
		}
		glPopMatrix();
		glEnable(GL_LIGHTING);
	//	qDebug() << nm << " is displayed - " << glList;
	}
}

bool vplane::define()
{
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	//glColor3f(0.0f, 0.0f, 1.0f);
	//glLoadName((GLuint)ID());
	glBegin(GL_QUADS);
	{
		glVertex3f(p0[0], p0[1], p0[2]);
		glVertex3f(p1[0], p1[1], p1[2]);
		glVertex3f(p2[0], p2[1], p2[2]);
		glVertex3f(p3[0], p3[1], p3[2]);
		//glVertex3f(p0[0], p0[1], p0[2]);
	}
	glEnd();
	glEndList();
	//qDebug() << "glList - " << glList;
	glHiList = glGenLists(1);
	glNewList(glHiList, GL_COMPILE);
	glBegin(GL_QUADS);
	{
		glVertex3f(p0[0], p0[1], p0[2]);
		glVertex3f(p1[0], p1[1], p1[2]);
		glVertex3f(p2[0], p2[1], p2[2]);
		glVertex3f(p3[0], p3[1], p3[2]);
		//glVertex3f(p0[0], p0[1], p0[2]);
	}
	glEnd();
	glEndList();
	return true;
}