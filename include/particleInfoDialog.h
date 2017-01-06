#ifndef PARTICLEINFODIALOG_H
#define PARTICLEINFODIALOG_H

#include <QDialog>
#include <QTextStream>
#include <string>
#include "vparticles.h"

namespace Ui {
	class PInfoDialog;
}

class particleInfoDialog : public QDialog
{
	Q_OBJECT

public:
	explicit particleInfoDialog(QWidget *parent = 0);
	~particleInfoDialog();

//	void bindingParticleViewer(parview::Object* par);
	void updateParticleInfo(unsigned int id);

	private slots:
	void sliderSlot();
	void pidLineEditSlot();
	void buttonSlot();

private:
//	parview::particles* parsys;
	Ui::PInfoDialog *ui;
};

#endif // NAVIGATORDLG_H