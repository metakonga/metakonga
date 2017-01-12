#include "vcube.h"
#include <QTextStream>

vcube::vcube()
	: vobject()
{
	setIndexList();
}

vcube::vcube(QString& _name)
	: vobject(_name)
{
	setIndexList();
}

vcube::vcube(QTextStream& in)
	: vobject()
{
	setIndexList();
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

	display = define();
	return true;
}

bool vcube::makeCubeGeometry(QString& _name, tRoll _tr, tMaterial _tm, VEC3F& _mp, VEC3F& _sz)
{
	vertice[0] = _mp.x;		   vertice[1] = _mp.y;		   vertice[2] = _mp.z;
	vertice[3] = _mp.x;		   vertice[4] = _mp.y + _sz.y; vertice[5] = _mp.z;
	vertice[6] = _mp.x;		   vertice[7] = _mp.y;		   vertice[8] = _mp.z + _sz.z;
	vertice[9] = _mp.x;		   vertice[10] = _mp.y + _sz.y; vertice[11] = _mp.z + _sz.z;
	vertice[12] = _mp.x + _sz.x; vertice[13] = _mp.y;		   vertice[14] = _mp.z + _sz.z;
	vertice[15] = _mp.x + _sz.x; vertice[16] = _mp.y + _sz.y; vertice[17] = _mp.z + _sz.z;
	vertice[18] = _mp.x + _sz.x; vertice[19] = _mp.y;		   vertice[20] = _mp.z;
	vertice[21] = _mp.x + _sz.x; vertice[22] = _mp.y + _sz.y; vertice[23] = _mp.z;
	this->define();
	return true;
}

void vcube::draw(GLenum eMode)
{
	if (display){
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glPushMatrix();
		glDisable(GL_LIGHTING);
		if (eMode == GL_SELECT) glLoadName((GLuint)ID());
		glCallList(glList);
		glPopMatrix();
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
		glVertex3f(vertice[id[3] * 3 + 0], vertice[id[3] * 3 + 1], vertice[id[3] * 3 + 2]);
		glVertex3f(vertice[id[2] * 3 + 0], vertice[id[2] * 3 + 1], vertice[id[2] * 3 + 2]);
		glVertex3f(vertice[id[1] * 3 + 0], vertice[id[1] * 3 + 1], vertice[id[1] * 3 + 2]);
		glVertex3f(vertice[id[0] * 3 + 0], vertice[id[0] * 3 + 1], vertice[id[0] * 3 + 2]);
	}

	glEnd();
	glEndList();

	return true;
}