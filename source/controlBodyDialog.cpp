#include "controlBodyDialog.h"

controlBodyTool::controlBodyTool(pointMass* pm, QWidget* parent /* = NULL */)
	: t(pm)
	, QDialog(parent)
{
	connect(TB_UPArrow, SLOT(clicked()), this, SIGNAL(up()));
	connect(TB_RIGHTArrow, SLOT(clicked()), this, SIGNAL(right()));
	connect(TB_BOTTOMArrow, SLOT(clicked()), this, SIGNAL(bottom()));
	connect(TB_LEFTArrow, SLOT(clicked()), this, SIGNAL(left()));
	connect(TB_XRotation, SLOT(clicked()), this, SIGNAL(xRot()));
	connect(TB_YRotation, SLOT(clicked()), this, SIGNAL(yRot()));
	connect(TB_ZRotation, SLOT(clicked()), this, SIGNAL(zRot()));
}

controlBodyTool::~controlBodyTool()
{

}

void controlBodyTool::up()
{

}

void controlBodyTool::bottom()
{

}

void controlBodyTool::left()
{

}

void controlBodyTool::right()
{

}

void controlBodyTool::xRot()
{

}

void controlBodyTool::yRot()
{

}

void controlBodyTool::zRot()
{

}

