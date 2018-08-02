/********************************************************************************
** Form generated from reading UI file 'contactPair.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONTACTPAIR_H
#define UI_CONTACTPAIR_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_ContactPair
{
public:
    QTabWidget *TB_METHOD;
    QWidget *T_DHS;
    QGroupBox *GB_ContactPair;
    QWidget *widget;
    QGridLayout *gridLayout;
    QLabel *L_FirstObject;
    QComboBox *CB_FirstObject;
    QLabel *L_SecondObject;
    QComboBox *CB_SecondObject;
    QGroupBox *GB_Parameter;
    QWidget *widget1;
    QGridLayout *gridLayout_2;
    QLabel *L_Restitution;
    QLineEdit *LE_Restitution;
    QLabel *L_StiffnessRatio;
    QLineEdit *LE_StiffnessRatio;
    QLabel *L_Friction;
    QLineEdit *LE_Friction;
    QWidget *tab_2;
    QWidget *widget2;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QWidget *widget3;
    QHBoxLayout *horizontalLayout_2;
    QLabel *L_Name;
    QLineEdit *LE_Name;

    void setupUi(QDialog *DLG_ContactPair)
    {
        if (DLG_ContactPair->objectName().isEmpty())
            DLG_ContactPair->setObjectName(QStringLiteral("DLG_ContactPair"));
        DLG_ContactPair->resize(344, 324);
        TB_METHOD = new QTabWidget(DLG_ContactPair);
        TB_METHOD->setObjectName(QStringLiteral("TB_METHOD"));
        TB_METHOD->setGeometry(QRect(10, 50, 328, 235));
        T_DHS = new QWidget();
        T_DHS->setObjectName(QStringLiteral("T_DHS"));
        GB_ContactPair = new QGroupBox(T_DHS);
        GB_ContactPair->setObjectName(QStringLiteral("GB_ContactPair"));
        GB_ContactPair->setGeometry(QRect(10, 10, 301, 80));
        widget = new QWidget(GB_ContactPair);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 20, 281, 48));
        gridLayout = new QGridLayout(widget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_FirstObject = new QLabel(widget);
        L_FirstObject->setObjectName(QStringLiteral("L_FirstObject"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(L_FirstObject->sizePolicy().hasHeightForWidth());
        L_FirstObject->setSizePolicy(sizePolicy);

        gridLayout->addWidget(L_FirstObject, 0, 0, 1, 1);

        CB_FirstObject = new QComboBox(widget);
        CB_FirstObject->setObjectName(QStringLiteral("CB_FirstObject"));

        gridLayout->addWidget(CB_FirstObject, 0, 1, 1, 1);

        L_SecondObject = new QLabel(widget);
        L_SecondObject->setObjectName(QStringLiteral("L_SecondObject"));
        sizePolicy.setHeightForWidth(L_SecondObject->sizePolicy().hasHeightForWidth());
        L_SecondObject->setSizePolicy(sizePolicy);

        gridLayout->addWidget(L_SecondObject, 1, 0, 1, 1);

        CB_SecondObject = new QComboBox(widget);
        CB_SecondObject->setObjectName(QStringLiteral("CB_SecondObject"));

        gridLayout->addWidget(CB_SecondObject, 1, 1, 1, 1);

        GB_Parameter = new QGroupBox(T_DHS);
        GB_Parameter->setObjectName(QStringLiteral("GB_Parameter"));
        GB_Parameter->setGeometry(QRect(10, 100, 301, 103));
        widget1 = new QWidget(GB_Parameter);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(10, 20, 281, 74));
        gridLayout_2 = new QGridLayout(widget1);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Restitution = new QLabel(widget1);
        L_Restitution->setObjectName(QStringLiteral("L_Restitution"));

        gridLayout_2->addWidget(L_Restitution, 0, 0, 1, 1);

        LE_Restitution = new QLineEdit(widget1);
        LE_Restitution->setObjectName(QStringLiteral("LE_Restitution"));

        gridLayout_2->addWidget(LE_Restitution, 0, 1, 1, 1);

        L_StiffnessRatio = new QLabel(widget1);
        L_StiffnessRatio->setObjectName(QStringLiteral("L_StiffnessRatio"));

        gridLayout_2->addWidget(L_StiffnessRatio, 1, 0, 1, 1);

        LE_StiffnessRatio = new QLineEdit(widget1);
        LE_StiffnessRatio->setObjectName(QStringLiteral("LE_StiffnessRatio"));

        gridLayout_2->addWidget(LE_StiffnessRatio, 1, 1, 1, 1);

        L_Friction = new QLabel(widget1);
        L_Friction->setObjectName(QStringLiteral("L_Friction"));

        gridLayout_2->addWidget(L_Friction, 2, 0, 1, 1);

        LE_Friction = new QLineEdit(widget1);
        LE_Friction->setObjectName(QStringLiteral("LE_Friction"));

        gridLayout_2->addWidget(LE_Friction, 2, 1, 1, 1);

        TB_METHOD->addTab(T_DHS, QString());
        GB_Parameter->raise();
        GB_ContactPair->raise();
        GB_ContactPair->raise();
        GB_Parameter->raise();
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        TB_METHOD->addTab(tab_2, QString());
        widget2 = new QWidget(DLG_ContactPair);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(150, 290, 181, 25));
        horizontalLayout = new QHBoxLayout(widget2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(widget2);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(widget2);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout->addWidget(PB_Cancle);

        widget3 = new QWidget(DLG_ContactPair);
        widget3->setObjectName(QStringLiteral("widget3"));
        widget3->setGeometry(QRect(10, 20, 321, 22));
        horizontalLayout_2 = new QHBoxLayout(widget3);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Name = new QLabel(widget3);
        L_Name->setObjectName(QStringLiteral("L_Name"));

        horizontalLayout_2->addWidget(L_Name);

        LE_Name = new QLineEdit(widget3);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));

        horizontalLayout_2->addWidget(LE_Name);


        retranslateUi(DLG_ContactPair);

        QMetaObject::connectSlotsByName(DLG_ContactPair);
    } // setupUi

    void retranslateUi(QDialog *DLG_ContactPair)
    {
        DLG_ContactPair->setWindowTitle(QApplication::translate("DLG_ContactPair", "Make contact pair", Q_NULLPTR));
        GB_ContactPair->setTitle(QApplication::translate("DLG_ContactPair", "Contact pair", Q_NULLPTR));
        L_FirstObject->setText(QApplication::translate("DLG_ContactPair", "First object", Q_NULLPTR));
        L_SecondObject->setText(QApplication::translate("DLG_ContactPair", "Second object", Q_NULLPTR));
        GB_Parameter->setTitle(QApplication::translate("DLG_ContactPair", "Parameter", Q_NULLPTR));
        L_Restitution->setText(QApplication::translate("DLG_ContactPair", "Restitution", Q_NULLPTR));
        L_StiffnessRatio->setText(QApplication::translate("DLG_ContactPair", "Stiffness ratio", Q_NULLPTR));
        L_Friction->setText(QApplication::translate("DLG_ContactPair", "Friction", Q_NULLPTR));
        TB_METHOD->setTabText(TB_METHOD->indexOf(T_DHS), QApplication::translate("DLG_ContactPair", "DHS", Q_NULLPTR));
        TB_METHOD->setTabText(TB_METHOD->indexOf(tab_2), QApplication::translate("DLG_ContactPair", "Tab 2", Q_NULLPTR));
        PB_Ok->setText(QApplication::translate("DLG_ContactPair", "Ok", Q_NULLPTR));
        PB_Cancle->setText(QApplication::translate("DLG_ContactPair", "Cancle", Q_NULLPTR));
        L_Name->setText(QApplication::translate("DLG_ContactPair", "Name", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_ContactPair: public Ui_DLG_ContactPair {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONTACTPAIR_H
