#include "vmarker.h"
#include "qgl.h"

float vmarker::scale = 1.0;

vmarker::vmarker()
	: vobject()
//	, scale(1.0)
	, markerScaleFlag(true)
{
	
}

vmarker::vmarker(QString& n, bool mcf)
	: vobject(n)
	///*,*/ scale(1.0)
	, markerScaleFlag(mcf)
{

}

vmarker::~vmarker()
{
	glDeleteLists(glList, 1);
}

void vmarker::draw(GLenum eMode)
{
	if (display)
	{
		glPushMatrix();
		if (eMode == GL_SELECT){
			glLoadName((GLuint)ID());
		}
		glDisable(GL_LIGHTING);
		if(markerScaleFlag)
			glScalef(scale, scale, scale);
		glTranslated(cpos.x, cpos.y, cpos.z);
		glRotated(cang.x / 16, 1.0, 0.0, 0.0); 
		glRotated(cang.y / 16, 0.0, 1.0, 0.0);
		glRotated(cang.z / 16, 0.0, 0.0, 1.0);

		glCallList(glList);
		glEnable(GL_LIGHTING);
		glPopMatrix();
	}	
}

bool vmarker::define(VEC3D p)
{
	cpos = pos0 = p;
	float icon_scale = 0.08;
	glList = glGenLists(1);
	glNewList(glList, GL_COMPILE);
	glShadeModel(GL_FLAT);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	{
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(icon_scale*1.0f, 0.0f, 0.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		glVertex3f(icon_scale*1.5f, 0.0f, 0.0f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			glVertex3f(icon_scale*1.0f, cos(rad)*icon_scale*0.15f, sin(rad)*icon_scale*0.15f);
		}
	}
	glEnd();

	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	{
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, icon_scale*1.0f, 0.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		glVertex3f(0.0f, icon_scale*1.5f, 0.0f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			glVertex3f(cos(rad)*icon_scale*0.15f, icon_scale*1.0f, sin(rad)*icon_scale*0.15f);
		}
	}
	glEnd();

	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_LINES);
	{
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, 0.0f, icon_scale*1.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		glVertex3f(0.0f, 0.0f, icon_scale*1.5f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			glVertex3f(cos(rad)*icon_scale*0.15f, sin(rad)*icon_scale*0.15f, icon_scale*1.0f);
		}
	}
	glEnd();

	glEndList();
	return true;
}