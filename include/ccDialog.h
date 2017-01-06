#ifndef CCDIALOG_H
#define CCDIALOG_H

/*#include "Object.h"*/
#include "mphysics_types.h"
#include "mphysics_numeric.h"
#include <QDialog>

QT_BEGIN_NAMESPACE
class QComboBox;
class QLineEdit;
class QTextStream;
QT_END_NAMESPACE

class modeler;
class object;
class collision;

class ccDialog : public QDialog
{
	Q_OBJECT

public:
	ccDialog();
	~ccDialog();

	//contact_coefficient_t CalcContactCoefficient(float ir, float jr, float im, float jm);
	collision* callDialog(modeler *md);
//	void SaveConstant(QTextStream& out);
//	void SetDataFromFile(QTextStream& in);

	bool isDialogOk;

	QLineEdit *LEName;
	QComboBox *list1;
	QComboBox *list2;
	QLineEdit *LErest;
	//QLineEdit *LEDamping;
	QLineEdit *LEfric;
	QLineEdit *LECohesion;
	QLineEdit *LEratio;

private slots:
	void clickOk();
	void clickCancel();
};


#endif