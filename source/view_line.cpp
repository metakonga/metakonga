#include "view_line.h"
#include <QtWidgets>

using namespace parview;

// unsigned int line::nline = 0;
// 
// line::line()
// 	: Object(LINE)
// 	, lineDialog(NULL)
// {
// 
// }
// 
// line::~line()
// {
// 	if (lineDialog) delete lineDialog; lineDialog = NULL;
// }
// 
// void line::setLineData(QFile& pf)
// {
// 	type = LINE;
// 	int name_size = 0;
// 	char nm[256] = { 0, };
// 	pf.read((char*)&name_size, sizeof(int));
// 	pf.read((char*)nm, sizeof(char)*name_size);
// 	name.sprintf("%s", nm);
// 	double sp[3];
// 	double ep[3];
// 
// 	pf.read((char*)&sp, sizeof(double)*3);
// 	pf.read((char*)&ep, sizeof(double)*3);
// 
// 	startPoint[0] = static_cast<float>(sp[0]);
// 	startPoint[1] = static_cast<float>(sp[1]);
// 	startPoint[2] = static_cast<float>(sp[2]);
// 
// 	endPoint[0] = static_cast<float>(ep[0]);
// 	endPoint[1] = static_cast<float>(ep[1]);
// 	endPoint[2] = static_cast<float>(ep[2]);
// }
// 
// bool line::callDialog(DIALOGTYPE dt)
// {
// 	QString nm;
// 	if (!lineDialog){
// 		QTextStream(&nm) << "Line_" << nline;
// 		lineDialog = new QDialog;
// 		QLabel *LName = new QLabel("Name");
// 		QLabel *LPa = new QLabel("Point a");
// 		QLabel *LPb = new QLabel("Point b");
// 		LEName = new QLineEdit(nm);
// 		LEPa = new QLineEdit;
// 		LEPb = new QLineEdit;
// 		QGridLayout *lineLayout = new QGridLayout;
// 		QPushButton *PBOk = new QPushButton("OK");
// 		QPushButton *PBCancel = new QPushButton("Cancel");
// 		connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
// 		connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
// 		lineLayout->addWidget(LMaterial, 0, 0);
// 		lineLayout->addWidget(CBMaterial, 0, 1, 1, 2);
// 		lineLayout->addWidget(LName, 1, 0);		lineLayout->addWidget(LEName, 1, 1, 1, 2);
// 		lineLayout->addWidget(LPa, 2, 0);		lineLayout->addWidget(LEPa, 2, 1, 1, 2);
// 		lineLayout->addWidget(LPb, 3, 0);		lineLayout->addWidget(LEPb, 3, 1, 1, 2);
// 		lineLayout->addWidget(PBOk, 6, 0);
// 		lineLayout->addWidget(PBCancel, 6, 1);
// 		lineDialog->setLayout(lineLayout);
// 	}
// 	lineDialog->exec();
// 	return isDialogOk ? true : false;
// }
// 
// void line::Click_ok()
// {
// 	if (LEPa->text().isEmpty() || LEPb->text().isEmpty()){
// 		msgBox("There is empty data!!", QMessageBox::Critical);
// 		return;
// 	}
// 
// 	QStringList ss;// = LEPa->text().split(",");
// 	if (LEPa->text().split(",").size() == 3)
// 		ss = LEPa->text().split(",");
// 	else
// 		if (LEPa->text().split(" ").size() == 3)
// 			ss = LEPa->text().split(" ");
// 		else
// 			if (LEPa->text().split(", ").size() == 3)
// 				ss = LEPa->text().split(", ");
// 			else {
// 				msgBox("Point a is wrong data.", QMessageBox::Critical);
// 				return;
// 			}
// 	startPoint[0] = ss.at(0).toFloat(); startPoint[1] = ss.at(1).toFloat(); startPoint[2] = ss.at(2).toFloat();
// 
// 			if (LEPb->text().split(",").size() == 3)
// 				ss = LEPb->text().split(",");
// 			else
// 				if (LEPb->text().split(" ").size() == 3)
// 					ss = LEPb->text().split(" ");
// 				else
// 					if (LEPb->text().split(", ").size() == 3)
// 						ss = LEPb->text().split(", ");
// 					else{
// 						msgBox("Point b is wrong data.", QMessageBox::Critical);
// 						return;
// 					}
// 	endPoint[0] = ss.at(0).toFloat(); endPoint[1] = ss.at(1).toFloat(); endPoint[2] = ss.at(2).toFloat();
// 
// 	mtype = material_str2enum(CBMaterial->currentText().toStdString());
// 	material = getMaterialConstant(mtype);
// 
// 	name = LEName->text();
// 
// 	nline++;
// 	lineDialog->close();
// 	delete lineDialog; lineDialog = NULL;
// 	isDialogOk = true;
// }
// 
// void line::Click_cancel()
// {
// 	lineDialog->close();
// 	delete lineDialog; lineDialog = NULL;
// 	isDialogOk = false;
// }
// 
// void line::draw(GLenum eMode)
// {
// 	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
// 	glPushMatrix();
// 	glDisable(GL_LIGHTING);
// 	glColor3fv(Object::color);
// 	glCallList(glList);
// 	glPopMatrix();
// }
// 
// bool line::define(void* tg)
// {
// 	glList = glGenLists(1);
// 	glNewList(glList, GL_COMPILE);
// 	glShadeModel(GL_SMOOTH);
// 	glColor3f(1.0f, 0.0f, 0.0f);
// 
// 	glBegin(GL_LINES);
// 	{
// 		glVertex3f(startPoint[0], startPoint[1], startPoint[2]);
// 		glVertex3f(endPoint[0], endPoint[1], endPoint[2]);
// 	}
// 	glEnd();
// 	glEndList();
// 	return true;
// }
// 
// void line::saveCurrentData(QFile& pf)
// {
// 
// }
// 
// void line::SaveObject(QTextStream& out)
// {
// 	out << "OBJECT" << " " << "LINE" << " " << name << " " << (int)roll << "\n";
// 	out << (int)mtype << "\n";
// 	out << startPoint[0] << " " << startPoint[1] << " " << startPoint[2] << "\n";
// 	out << endPoint[0] << " " << endPoint[1] << " " << endPoint[2] << "\n";
// }
// 
// void line::SetDataFromFile(QTextStream& in)
// {
// 	int _roll, _mtype;
// 	in >> name >> _roll
// 		>> _mtype
// 		>> startPoint[0] >> startPoint[1] >> startPoint[2]
// 		>> endPoint[0] >> endPoint[1] >> endPoint[2];
// 	roll = (ObjectRoll)_roll;
// 	mtype = (material_type)_mtype;
// 	material = getMaterialConstant(mtype);
// 	if (roll == ROLL_PARTICLE)
// 		Object::isHide = true;
// }
// 
// void line::updateDataFromFile(QFile& pf, unsigned int fdtype)
// {
// 
// }