#include "view_bondData.h"

using namespace parview;

// bondData::bondData()
//	: Object()
//	, bds(NULL)
//	, size(0)
//{
//
//}
//
//bondData::~bondData()
//{
//	if (bds)
//		delete[] bds;
//	bds = NULL;
//}
//
//void bondData::setBondData(QFile& pf)
//{
//	pf.read((char*)&size, sizeof(unsigned int));
//	bds = new bond_data[size];
//	pf.read((char*)bds, sizeof(bond_data)*size);
//}
//
//void bondData::draw(GLenum eMode)
//{
//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//	glPushMatrix();
//	glDisable(GL_LIGHTING);
//	//glColor3fv(Object::color);
//	glCallList(glList);
//	glPopMatrix();
//}
//
//void bondData::define(void* tg)
//{
//	glList = glGenLists(1);
//	glNewList(glList, GL_COMPILE);
//	glShadeModel(GL_SMOOTH);
//	glColor3f(1.0f, 0.0f, 0.0f);
//
//	for (unsigned int i = 0; i < size; i++){
//		if (bds[i].broken){
//			glBegin(GL_LINES);
//			{
//				glVertex3f(bds[i].sp.x, bds[i].sp.y, bds[i].sp.z);
//				glVertex3f(bds[i].ep.x, bds[i].ep.y, bds[i].ep.z);
//			}
//		}
//	}
//	
//	glEnd();
//	glEndList();
//}
//
//void bondData::saveCurrentData(QFile& pf)
//{
//
//}