#ifndef CONTACTPAIRDIALOG_H
#define CONTACTPAIRDIALOG_H

#include "ui_contactPair.h"
#include "types.h"

class contactPairDialog : public QDialog, private Ui::DLG_ContactPair
{
 	Q_OBJECT

public:
	contactPairDialog(QWidget* parent = NULL);
	~contactPairDialog();

	QString name;
	int method;

	QString firstObj;
	QString secondObj;

	double restitution;
	double stiffnessRatio;
	double friction;

	void setObjectLists(QStringList& list);

	private slots:
	void changeComboBox(int);
	void click_ok();
	void click_cancle();
};

#endif
// 
// /*#include "Object.h"*/
// #include "types.h"
// #include "vectorTypes.h"
// #include <QDialog>
// 
// QT_BEGIN_NAMESPACE
// class QComboBox;
// class QLineEdit;
// class QTextStream;
// QT_END_NAMESPACE
// 
// class modeler;
// class object;
// class collision;
// 
// class ccDialog : public QDialog
// {
// 	Q_OBJECT
// 
// public:
// 	ccDialog();
// 	~ccDialog();
// 
// 	//contact_coefficient_t CalcContactCoefficient(float ir, float jr, float im, float jm);
// 	collision* callDialog(modeler *md);
// //	void SaveConstant(QTextStream& out);
// //	void SetDataFromFile(QTextStream& in);
// 
// 	bool isDialogOk;
// 
// 	QLineEdit *LEName;
// 	QComboBox *list1;
// 	QComboBox *list2;
// 	QLineEdit *LErest;
// 	//QLineEdit *LEDamping;
// 	QLineEdit *LEfric;
// 	QLineEdit *LERollingFriction;
// 	QLineEdit *LECohesion;
// 
// private slots:
// 	void clickOk();
// 	void clickCancel();
// };
// 
// 
// #endif