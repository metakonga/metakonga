#include "motionConditionDialog.h"

motionConditionDialog::motionConditionDialog(QWidget* parent /* = NULL */)
	: QDialog(parent)
	, st(0.0)
	, et(0.0)
	, cv(0.0)
	, isConstantVelocity(false)
{
	setupUi(this);
	LE_StartTime->setText("0.0");
	LE_EndTime->setText("0.0");
	LE_Value->setText("0.0");
	LE_Direction->setText("0.0 0.0 0.0");

	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(Click_cancel()));
}

motionConditionDialog::~motionConditionDialog()
{

}

void motionConditionDialog::setName(QString& nm)
{
	LE_Name->setText(nm);
}

void motionConditionDialog::Click_ok()
{
	if (RB_ConstantVelocity->isChecked())
	{
		st = LE_StartTime->text().toDouble();
		et = LE_EndTime->text().toDouble();
		cv = LE_Value->text().toDouble();
		QStringList sl = LE_Direction->text().split(" ");
		ux = sl.at(0).toDouble();
		uy = sl.at(1).toDouble();
		uz = sl.at(2).toDouble();
	}
	else
	{
		Click_cancel();
	}
	this->close();
	this->setResult(QDialog::Accepted);
}

void motionConditionDialog::Click_cancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
}
