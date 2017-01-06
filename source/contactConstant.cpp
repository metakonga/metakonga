#include "contactConstant.h"
#include "Object.h"
#include <QtWidgets>

bool parview::contactConstant::callDialog(QStringList& strList)
{
	if (!ccdialog)
	{
		ccdialog = new QDialog;
		list1 = new QComboBox; list1->addItems(strList);
		list2 = new QComboBox; list2->addItems(strList);
		QLabel *LList1 = new QLabel("First Object");
		QLabel *LList2 = new QLabel("Second Object");
		QLabel *Lrest = new QLabel("Restitution");
		QLabel *LRatio = new QLabel("Stiffness ratio");
		QLabel *LFric = new QLabel("Friction");
		LErest = new QLineEdit;
		LEfric = new QLineEdit;
		LEratio = new QLineEdit;

		QGridLayout *ccLayout = new QGridLayout;
		QPushButton *PBOk = new QPushButton("OK");
		QPushButton *PBCancel = new QPushButton("Cancel");
		connect(PBOk, SIGNAL(clicked()), this, SLOT(clickOk()));
		connect(PBCancel, SIGNAL(clicked()), this, SLOT(clickCancel()));
		ccLayout->addWidget(LList1, 0, 0);	ccLayout->addWidget(list1, 0, 1, 1, 2);
		ccLayout->addWidget(LList2, 1, 0);	ccLayout->addWidget(list2, 1, 1, 1, 2);
		ccLayout->addWidget(Lrest, 2, 0);	ccLayout->addWidget(LErest, 2, 1, 1, 2);
		ccLayout->addWidget(LRatio, 3, 0);	ccLayout->addWidget(LEratio, 3, 1, 1, 2);
		ccLayout->addWidget(LFric, 4, 0);	ccLayout->addWidget(LEfric, 4, 1, 1, 2);
		ccLayout->addWidget(PBOk, 5, 0);
		ccLayout->addWidget(PBCancel, 5, 1);
		ccdialog->setLayout(ccLayout);
	}
	ccdialog->exec();
	
	return isDialogOk;
}

void parview::contactConstant::clickOk()
{
// 	if (list1->currentText() == list2->currentText()){
// 		if (list1->currentText() != "particles" && list2->currentText() != "particles"){
// 			Object::msgBox("You selected same object.", QMessageBox::Critical);
// 			return;
// 		}			
// 	}
// 		
// 	if (LErest->text().isEmpty() || LEfric->text().isEmpty() || LEratio->text().isEmpty()){
// 		Object::msgBox("There is empty input.", QMessageBox::Critical);
// 		return;
// 	}
// 		
// 	
// 	obj_si = list1->currentText();
// 	obj_sj = list2->currentText();
// 	restitution = LErest->text().toFloat();
// 	friction = LEfric->text().toFloat();
// 	stiff_ratio = LEratio->text().toFloat();
// 	ccdialog->close();
// 	delete ccdialog; ccdialog = NULL;
// 
// 	isDialogOk = true;
}

void parview::contactConstant::clickCancel()
{
	ccdialog->close();
	delete ccdialog; ccdialog = NULL;
	isDialogOk = false;
}

void parview::contactConstant::SaveConstant(QTextStream& out)
{
	out << "CONTACT_CONSTANT" << " " << obj_si << " " << obj_sj << "\n";
	out << restitution << " " << friction << " " << stiff_ratio << "\n";
}

void parview::contactConstant::SetDataFromFile(QTextStream& in)
{
	in >> obj_si >> obj_sj >> restitution >> friction >> stiff_ratio;
}

// contact_coefficient_t parview::contactConstant::CalcContactCoefficient(float ir, float jr, float im, float jm)
// {
// // 	contact_coefficient_t cct;
// // 	float em = jm ? (im * jm) / (im + jm) : im;
// // 	float er = jr ? (ir * jr) / (ir + jr) : ir;
// // 	float eym = (obj_i->Material().youngs * obj_j->Material().youngs) / (obj_i->Material().youngs*(1 - obj_j->Material().poisson*obj_j->Material().poisson) + obj_j->Material().youngs*(1 - obj_i->Material().poisson*obj_i->Material().poisson));
// // 	float beta = (M_PI / log(restitution));
// // 	cct.kn = (4.0f / 3.0f)*sqrt(er)*eym;
// // 	cct.vn = sqrt((4.0f*em * cct.kn) / (1 + beta * beta));
// // 	cct.ks = cct.kn * stiff_ratio;
// // 	cct.vs = cct.vn * stiff_ratio;
// // 	cct.mu = friction;
// // 	return cct;
// }