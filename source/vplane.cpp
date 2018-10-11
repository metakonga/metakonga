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
	p0[0] = (float)pa.x; p0[1] = (float)pa.y; p0[2] = (float)pa.z;
	p1[0] = (float)pb.x; p1[1] = (float)pb.y; p1[2] = (float)pb.z;
	p2[0] = (float)pc.x; p2[1] = (float)pc.y; p2[2] = (float)pc.z;
	p3[0] = (float)pd.x; p3[1] = (float)pd.y; p3[2] = (float)pd.z;
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