#include "importDialog.h"
#include "msgBox.h"
#include <QtWidgets>
#include "types.h"
#include "model.h"

importDialog::importDialog(QWidget* parent)
	: QDialog(parent)
	, file_path("")
{
	setupUi(this);
	CB_Type->addItems(getMaterialList());
	int mt = CB_Type->currentIndex();
	cmaterialType cmt = getMaterialConstant(mt);
	LE_Youngs->setText(QString("%1").arg(cmt.youngs));
	LE_PoissonRatio->setText(QString("%1").arg(cmt.poisson));
	LE_Density->setText(QString("%1").arg(cmt.density));
	LE_ShearModulus->setText(QString("%1").arg(cmt.shear));
	LE_Youngs->setReadOnly(true);
	LE_PoissonRatio->setReadOnly(true);
	LE_Density->setReadOnly(true);
	LE_ShearModulus->setReadOnly(true);
	connect(PB_FileBrowser, SIGNAL(clicked()), this, SLOT(click_browser()));
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(Click_cancel()));
	connect(CB_Type, SIGNAL(currentIndexChanged(int)), this, SLOT(changeComboBox(int)));
}

importDialog::~importDialog()
{

}

void importDialog::click_browser()
{
	QString dir = model::path;
	QString fileName = QFileDialog::getOpenFileName(
		this, tr("Import"), dir, "MilkShape 3D ASCII (*.txt);;All files (*.*)");
	LE_FilePath->setText(fileName);
}

void importDialog::changeComboBox(int idx)
{
	if (idx == USER_INPUT)
	{
		LE_Youngs->setReadOnly(false);
		LE_PoissonRatio->setReadOnly(false);
		LE_Density->setReadOnly(false);
		LE_ShearModulus->setReadOnly(false);
	}
	else
	{
		cmaterialType cmt;
		cmt = getMaterialConstant(idx);
		LE_Youngs->setText(QString("%1").arg(cmt.youngs));
		LE_PoissonRatio->setText(QString("%1").arg(cmt.poisson));
		LE_Density->setText(QString("%1").arg(cmt.density));
		LE_ShearModulus->setText(QString("%1").arg(cmt.shear));
		LE_Youngs->setReadOnly(true);
		LE_PoissonRatio->setReadOnly(true);
		LE_Density->setReadOnly(true);
		LE_ShearModulus->setReadOnly(true);
	}
}

void importDialog::Click_ok()
{
	file_path = LE_FilePath->text();
	type = CB_Type->currentIndex();
	youngs = LE_Youngs->text().toDouble();
	poisson = LE_PoissonRatio->text().toDouble();
	density = LE_Density->text().toDouble();
	shear = LE_ShearModulus->text().toDouble();
	this->close();
	this->setResult(QDialog::Accepted);
}

void importDialog::Click_cancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
}