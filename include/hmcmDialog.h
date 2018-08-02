// #ifndef HMCMDIALOG_H
// #define HMCMDIALOG_H
// 
// #include "ui_HMCM.h"
// 
// class modeler;
// 
// class hmcmDialog : public QDialog, private Ui::DLG_HMCM
// {
// 	Q_OBJECT
// 		
// public:
// 	hmcmDialog(QWidget* parent, modeler* md);
// 	~hmcmDialog();
// 
// private:
// 	modeler* md;
// 
// 	void setupDialog();
// 
// 	private slots:
// 	void check_cohesion();
// 	void check_stiffnessRatio();
// 	void click_ok();
// 	void click_cancle();
// };
// 
// #endif