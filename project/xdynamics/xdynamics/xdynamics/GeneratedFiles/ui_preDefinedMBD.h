/********************************************************************************
** Form generated from reading UI file 'preDefinedMBD.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PREDEFINEDMBD_H
#define UI_PREDEFINEDMBD_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_PREDEFINEDMBD
{
public:
    QTreeWidget *TW_PreDefined_MBD;
    QWidget *widget;
    QHBoxLayout *ButtonLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;

    void setupUi(QDialog *DLG_PREDEFINEDMBD)
    {
        if (DLG_PREDEFINEDMBD->objectName().isEmpty())
            DLG_PREDEFINEDMBD->setObjectName(QStringLiteral("DLG_PREDEFINEDMBD"));
        DLG_PREDEFINEDMBD->resize(278, 246);
        TW_PreDefined_MBD = new QTreeWidget(DLG_PREDEFINEDMBD);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QStringLiteral("1"));
        TW_PreDefined_MBD->setHeaderItem(__qtreewidgetitem);
        TW_PreDefined_MBD->setObjectName(QStringLiteral("TW_PreDefined_MBD"));
        TW_PreDefined_MBD->setGeometry(QRect(10, 10, 256, 192));
        widget = new QWidget(DLG_PREDEFINEDMBD);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(50, 210, 211, 25));
        ButtonLayout = new QHBoxLayout(widget);
        ButtonLayout->setObjectName(QStringLiteral("ButtonLayout"));
        ButtonLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(widget);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        ButtonLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(widget);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        ButtonLayout->addWidget(PB_Cancle);


        retranslateUi(DLG_PREDEFINEDMBD);

        QMetaObject::connectSlotsByName(DLG_PREDEFINEDMBD);
    } // setupUi

    void retranslateUi(QDialog *DLG_PREDEFINEDMBD)
    {
        DLG_PREDEFINEDMBD->setWindowTitle(QApplication::translate("DLG_PREDEFINEDMBD", "Dialog", Q_NULLPTR));
        PB_Ok->setText(QApplication::translate("DLG_PREDEFINEDMBD", "Ok", Q_NULLPTR));
        PB_Cancle->setText(QApplication::translate("DLG_PREDEFINEDMBD", "Cancle", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_PREDEFINEDMBD: public Ui_DLG_PREDEFINEDMBD {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PREDEFINEDMBD_H
