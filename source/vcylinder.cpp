#include "vcylinder.h"
#include "vcontroller.h"
#include <QTextStream>

vcylinder::vcylinder()
	: vobject()
{

}

vcylinder::vcylinder(QString& _name)
	: vobject(_name)
{

}

vcylinder::vcylinder(QTextStream& in)
	: vobject()
{
	QString ch;
	in >> ch; nm = ch;//this->setObjectName(ch);
	//in >> idr >> idm;

	makeCylinderGeometry(in);
}

void vcylinder::draw(GLenum eMode)
{
	if (display){
		glDisable(GL_LIGHTING);
		glPushMatrix();
		if (vcontroller::getFrame() && outPos && outRot)
		{
			animationFrame(origin[0], origin[1], origin[2]);
// 			//glPushMatrix();
// 			unsigned int f = vcontroller::getFrame();
// 			glTranslated(outPos[f].x, outPos[f].y, outPos[f].z);
// 			VEC3D e = ep2e(outRot[f]);
// 			//e += VEC3D(ang[0], ang[1], ang[2]);
// 			double xi = (e.x * 180) / M_PI;
// 			double th = (e.y * 180) / M_PI;
// 			double ap = (e.z * 180) / M_PI;
// 			double diff = xi + ap;
// 			glRotated(xi/* - ang[0]*/, 0, 0, 1);
// 			glRotated(th/* - ang[1]*/, 1, 0, 0);
// 			glRotated(ap/* - ang[2]*/, 0, 0, 1);
// 			/*glPopMatrix();*/
// 			//glCallList(coord);
		}
		else{
// 			glTranslated(origin[0], origin[1], origin[2]);
// 			//glPushMatrix();
// 			glRotated(ang[0], 0, 0, 1);
// 			glRotated(ang[1], 1, 0, 0);
// 			glRotated(ang[2], 0, 0, 1);
			
			//glPopMatrix();
		}

		if (eMode == GL_SELECT) glLoadName((GLuint)ID());
		//glCallList(coord);
		glCallList(glList);
		glPopMatrix();
		glEnable(GL_LIGHTING);
	}
}

bool vcylinder::define()
{
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_FLAT);
	float angle = (float)(15 * (M_PI / 180));
	int iter = (int)(360 / 15);

	float h_len = length * 0.5f;
	VEC3F to = VEC3F(pb[0] - origin[0], pb[1] - origin[1], pb[2] - origin[2]);
	VEC3F u = to / to.length();
	//VEC3F t = to - to.dot(u) * u;
	double th = M_PI * 0.5;
	double ap = acos(u.z);
	double xi = asin(-u.y);

	if (ap > M_PI)
		ap = ap - M_PI;

	ang[0] = 0.f;// 180 * xi / M_PI;
	ang[1] = 0.f;// 180 * th / M_PI;
	ang[2] = 0.f;// 180 * ap / M_PI;

	EPD ep;
	ep.setFromEuler(xi, th, ap);

	glPushMatrix();
	glBegin(GL_TRIANGLE_FAN);
	{
		//VEC3D p = VEC3D( 0.f, length * 0.5f, 0.f );
		//glColor3f(0.0f, 0.f, 1.f);
	//	VEC3F p2_ = ep.A() * VEC3F(p2[0], p2[1], p2[2]);
		//glVertex3f(p2[0], p2[1], p2[2]);
		//p = ep.A() * p;
		glVertex3f(pb[0], pb[1], pb[2]);
		for (int i = 0; i < iter + 1; i++){
			float rad = angle * i;
			//glColor3f(i % 2, 0.f, i % 2 + 1.f);
			VEC3D q(sin(rad)*topRadius, cos(rad) * topRadius, -0.5f * length );
			//q = ep.A() * q;
			glVertex3f(/*origin[0] + */(float)q.x, /*origin[1] + */(float)q.y, /*origin[2] + */(float)q.z);	
		}
	}
	glEnd();
	glPopMatrix();
	glBegin(GL_TRIANGLE_FAN);
	{
		VEC3D p = VEC3D(0.f, -length * 0.5f, 0.f);
		//float p[3] = { 0.f, -length * 0.5f, 0.f };
		//glColor3f(0.f, 0.f, 1.f);
		//glVertex3f(p1[0], p1[1], p1[2]);
		//p = ep.A() * p;
		//glVertex3f(p.x, p.y, p.z);
		glVertex3f(pt[0], pt[1], pt[2]);
		for (int i = 0; i < iter + 1; i++){
			float rad = angle * i;
			//glColor3f(i % 2, 0.0f, i % 2 + 1.0f);
			VEC3D q(sin(-rad)*baseRadius, cos(-rad) * baseRadius, 0.5f * length);
			//q = ep.A() * q;
			glVertex3f(/*origin[0] + */q.x, /*origin[1] + */q.y, /*origin[2] +*/ q.z);
		}
	}
	glEnd();
	glBegin(GL_QUAD_STRIP);
	{
		for (int i = 0; i < iter + 1; i++){
			float rad = angle * i;
			VEC3D q1(sin(rad) * topRadius, cos(rad) * topRadius, length * 0.5);
			VEC3D q2(sin(rad) * baseRadius, cos(rad) * baseRadius, -length * 0.5);
			//q1 = ep.A() * q1;
			//q2 = ep.A() * q2;
			glVertex3f(/*origin[0] + */q2.x, /*origin[1] + */q2.y, /*origin[2] + */q2.z);
			glVertex3f(/*origin[0] + */q1.x, /*origin[1] + */q1.y, /*origin[2] + */q1.z);
		}
	}
	glEnd();
	glEndList();


	return true;
}

bool vcylinder::makeCylinderGeometry(QTextStream& in)
{
	return true;
}

bool vcylinder::makeCylinderGeometry(float br, float tr, float leng, VEC3F org, VEC3F _p1, VEC3F _p2)
{
	baseRadius = br;
	topRadius = tr;
	length = leng;
	origin[0] = org.x;  origin[1] = org.y; origin[2] = org.z;
	pb[0] = _p1.x; pb[1] = _p1.y; pb[2] = _p1.z;
	pt[0] = _p2.x; pt[1] = _p2.y; pt[2] = _p2.z;
	this->define();
	return true;
}