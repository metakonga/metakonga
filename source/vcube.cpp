#include "vcube.h"
#include <QTextStream>

vcube::vcube()
	: vobject()
{
	origin[0] = origin[1] = origin[2] = 0.f;
	setIndexList();
	setNormalList();
}

vcube::vcube(QString& _name)
	: vobject(_name)
{
	setIndexList();
	setNormalList();
}

vcube::vcube(QTextStream& in)
	: vobject()
{
	setIndexList();
	setNormalList();
	QString ch;
	in >> ch; nm = ch;//this->setObjectName(ch);
//	in >> idr >> idm;

	makeCubeGeometry(in);
}

void vcube::setIndexList()
{
	indice[0] = 0; indice[1] = 2; indice[2] = 4; indice[3] = 6;
	indice[4] = 1; indice[5] = 3; indice[6] = 2; indice[7] = 0;
	indice[8] = 5; indice[9] = 7; indice[10] = 6; indice[11] = 4;
	indice[12] = 3; indice[13] = 5; indice[14] = 4; indice[15] = 2;
	indice[16] = 7; indice[17] = 1; indice[18] = 0; indice[19] = 6;
	indice[20] = 1; indice[21] = 7; indice[22] = 5; indice[23] = 3;
}

void vcube::setNormalList()
{
	normal[0] = 0; normal[1] = -1.0; normal[2] = 0;
	normal[3] = -1.0; normal[4] = 0.0; normal[5] = 0.0;
	normal[6] = 1.0; normal[7] = 0.0; normal[8] = 0.0;
	normal[9] = 0.0; normal[10] = 0.0; normal[11] = 1.0;
	normal[12] = 0.0; normal[13] = 0.0; normal[14] = -1.0;
	normal[15] = 0.0; normal[16] = 1.0; normal[17] = 0.0;
}

bool vcube::makeCubeGeometry(QTextStream& in)
{
	float origin[3];
	float minPoint[3];
	float maxPoint[3];
	float size[3];
	in >> origin[0] >> origin[1] >> origin[2];
	in >> minPoint[0] >> minPoint[1] >> minPoint[2];
	in >> maxPoint[0] >> maxPoint[1] >> maxPoint[2];
	in >> size[0] >> size[1] >> size[2];

	vertice[0] = minPoint[0];		   vertice[1] = minPoint[1];		   vertice[2] = minPoint[2];
	vertice[3] = minPoint[0];		   vertice[4] = minPoint[1] + size[1]; vertice[5] = minPoint[2];
	vertice[6] = minPoint[0];		   vertice[7] = minPoint[1];		   vertice[8] = minPoint[2] + size[2];
	vertice[9] = minPoint[0];		   vertice[10] = minPoint[1] + size[1]; vertice[11] = minPoint[2] + size[2];
	vertice[12] = minPoint[0] + size[0]; vertice[13] = minPoint[1];		   vertice[14] = minPoint[2] + size[2];
	vertice[15] = minPoint[0] + size[0]; vertice[16] = minPoint[1] + size[1]; vertice[17] = minPoint[2] + size[2];
	vertice[18] = minPoint[0] + size[0]; vertice[19] = minPoint[1];		   vertice[20] = minPoint[2];
	vertice[21] = minPoint[0] + size[0]; vertice[22] = minPoint[1] + size[1]; vertice[23] = minPoint[2];
	origin[0] = minPoint[0] + size[0] * 0.5f;
	origin[1] = minPoint[1] + size[1] * 0.5f;
	origin[2] = minPoint[2] + size[2] * 0.5f;
	display = define();
	return true;
}

bool vcube::makeCubeGeometry(QString& _name, geometry_use _tr, material_type _tm, VEC3F& _mp, VEC3F& _sz)
{
	vertice[0] = _mp.x;		   vertice[1] = _mp.y;		   vertice[2] = _mp.z;
	vertice[3] = _mp.x;		   vertice[4] = _mp.y + _sz.y; vertice[5] = _mp.z;
	vertice[6] = _mp.x;		   vertice[7] = _mp.y;		   vertice[8] = _mp.z + _sz.z;
	vertice[9] = _mp.x;		   vertice[10] = _mp.y + _sz.y; vertice[11] = _mp.z + _sz.z;
	vertice[12] = _mp.x + _sz.x; vertice[13] = _mp.y;		   vertice[14] = _mp.z + _sz.z;
	vertice[15] = _mp.x + _sz.x; vertice[16] = _mp.y + _sz.y; vertice[17] = _mp.z + _sz.z;
	vertice[18] = _mp.x + _sz.x; vertice[19] = _mp.y;		   vertice[20] = _mp.z;
	vertice[21] = _mp.x + _sz.x; vertice[22] = _mp.y + _sz.y; vertice[23] = _mp.z;
	origin[0] = _mp.x + _sz.x * 0.5f;
	origin[1] = _mp.y + _sz.y * 0.5f;
	origin[2] = _mp.z + _sz.z * 0.5f;
	this->define();
	return true;
}

void vcube::draw(GLenum eMode)
{
	if (display){
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glPushMatrix();
		//glDisable(GL_LIGHTING);
		if (vcontroller::getFrame() && outPos && outRot)
			animationFrame(origin[0], origin[1], origin[2]);
		if (eMode == GL_SELECT){
			glLoadName((GLuint)ID());
		}
		//glLineWidth(10);
		
		/*glLineStipple(5, 0x5555);*/
		//glEnable(GL_LINE_STIPPLE);
		glColor3f(clr.redF(), clr.greenF(), clr.blueF());
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
		//glEnable(GL_LIGHTING);
	}
}

bool vcube::define()
{
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_SMOOTH);

	glBegin(GL_QUADS);
	for (int i(0); i < 6; i++){
		int *id = &indice[i * 4];
		glNormal3f(normal[i * 3 + 0], normal[i * 3 + 1], normal[i * 3 + 2]);
		glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
		glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
		glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
		glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	}
	glEnd();
	glEndList();


	glHiList = glGenLists(1);
	glNewList(glHiList, GL_COMPILE);
//	glColor3f(1.0, 0.0, 0.0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_LINE_LOOP);
	int *id = &indice[0];
	glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
	glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
	glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
	glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	glEnd(); 
	glBegin(GL_LINE_LOOP);
	id = &indice[5 * 4];
	glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
	glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
	glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
	glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(vertice[0], vertice[1], vertice[2]);
	glVertex3f(vertice[3], vertice[4], vertice[5]);

	glVertex3f(vertice[6], vertice[7], vertice[8]);
	glVertex3f(vertice[9], vertice[10], vertice[11]);

	glVertex3f(vertice[12], vertice[13], vertice[14]);
	glVertex3f(vertice[15], vertice[16], vertice[17]);

	glVertex3f(vertice[18], vertice[19], vertice[20]);
	glVertex3f(vertice[21], vertice[22], vertice[23]);
	glEnd();
	glEndList();
	return true;
}