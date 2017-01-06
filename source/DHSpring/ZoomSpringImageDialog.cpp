#include "ZoomSpringImageDialog.h"
#include "ui_zoomSpringImage.h"
#include <QTextStream>

ZoomSpringImageDialog::ZoomSpringImageDialog(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::SpringImageZoomDialog)

{
	ui->setupUi(this);
}

ZoomSpringImageDialog::~ZoomSpringImageDialog()
{

}

void ZoomSpringImageDialog::setValue()
{
	QString str;
	QTextStream(&str) << d; ui->LE_d->setText(str); str.clear();
	QTextStream(&str) << f_length; ui->LE_Free_Length->setText(str); str.clear();
	QTextStream(&str) << equip_length; ui->LE_Inst_Length->setText(str); str.clear();
	QTextStream(&str) << init_comp; ui->LE_Init_Comp->setText(str); str.clear();
	QTextStream(&str) << stroke; ui->LE_Stroke->setText(str); str.clear();
	QTextStream(&str) << max_comp; ui->LE_Max_Comp->setText(str); str.clear();
	QTextStream(&str) << inner_diameter; ui->LE_Inner_Diameter->setText(str); str.clear();
	QTextStream(&str) << outer_diameter; ui->LE_Outer_Diameter->setText(str); str.clear();
	QTextStream(&str) << middle_diameter; ui->LE_Middle_Diameter->setText(str); str.clear();
}