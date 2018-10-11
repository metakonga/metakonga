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
	bool ignore_condition;
	QString firstObj;
	QString secondObj;

	double restitution;
	double stiffnessRatio;
	double friction;
	double cohesion;

	double ignore_time;

	void setObjectLists(QStringList& list);
	void setComboBoxString(QString& f, QString& s);
	void setContactParameters(double r, double s, double f);
	void setIgnoreCondition(bool b, double t);

	private slots:
	void changeComboBox(int);
	void checkBoxSlot();
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