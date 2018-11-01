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
#include <QtWidgets/QCheckBox>
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
    QWidget *layoutWidget;
    QGridLayout *gridLayout;
    QLabel *L_FirstObject;
    QComboBox *CB_FirstObject;
    QLabel *L_SecondObject;
    QComboBox *CB_SecondObject;
    QGroupBox *GB_Parameter;
    QWidget *widget;
    QGridLayout *gridLayout_2;
    QLabel *L_Restitution;
    QLineEdit *LE_Restitution;
    QLabel *L_StiffnessRatio;
    QLineEdit *LE_StiffnessRatio;
    QLabel *L_Friction;
    QLineEdit *LE_Friction;
    QLabel *L_Cohesion;
    QLineEdit *LE_Cohesion;
    QWidget *tab_2;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QWidget *layoutWidget2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *L_Name;
    QLineEdit *LE_Name;
    QCheckBox *CHB_IgnoreCondition;
    QWidget *layoutWidget3;
    QHBoxLayout *horizontalLayout_3;
    QLabel *L_IgnoreTime;
    QLineEdit *LE_IgnoreTime;

    void setupUi(QDialog *DLG_ContactPair)
    {
        if (DLG_ContactPair->objectName().isEmpty())
            DLG_ContactPair->setObjectName(QStringLiteral("DLG_ContactPair"));
        DLG_ContactPair->resize(344, 382);
        TB_METHOD = new QTabWidget(DLG_ContactPair);
        TB_METHOD->setObjectName(QStringLiteral("TB_METHOD"));
        TB_METHOD->setGeometry(QRect(10, 50, 328, 261));
        T_DHS = new QWidget();
        T_DHS->setObjectName(QStringLiteral("T_DHS"));
        GB_ContactPair = new QGroupBox(T_DHS);
        GB_ContactPair->setObjectName(QStringLiteral("GB_ContactPair"));
        GB_ContactPair->setGeometry(QRect(10, 10, 301, 80));
        layoutWidget = new QWidget(GB_ContactPair);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(10, 20, 281, 48));
        gridLayout = new QGridLayout(layoutWidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_FirstObject = new QLabel(layoutWidget);
        L_FirstObject->setObjectName(QStringLiteral("L_FirstObject"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(L_FirstObject->sizePolicy().hasHeightForWidth());
        L_FirstObject->setSizePolicy(sizePolicy);

        gridLayout->addWidget(L_FirstObject, 0, 0, 1, 1);

        CB_FirstObject = new QComboBox(layoutWidget);
        CB_FirstObject->setObjectName(QStringLiteral("CB_FirstObject"));

        gridLayout->addWidget(CB_FirstObject, 0, 1, 1, 1);

        L_SecondObject = new QLabel(layoutWidget);
        L_SecondObject->setObjectName(QStringLiteral("L_SecondObject"));
        sizePolicy.setHeightForWidth(L_SecondObject->sizePolicy().hasHeightForWidth());
        L_SecondObject->setSizePolicy(sizePolicy);

        gridLayout->addWidget(L_SecondObject, 1, 0, 1, 1);

        CB_SecondObject = new QComboBox(layoutWidget);
        CB_SecondObject->setObjectName(QStringLiteral("CB_SecondObject"));

        gridLayout->addWidget(CB_SecondObject, 1, 1, 1, 1);

        GB_Parameter = new QGroupBox(T_DHS);
        GB_Parameter->setObjectName(QStringLiteral("GB_Parameter"));
        GB_Parameter->setGeometry(QRect(10, 100, 301, 131));
        widget = new QWidget(GB_Parameter);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 21, 281, 100));
        gridLayout_2 = new QGridLayout(widget);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Restitution = new QLabel(widget);
        L_Restitution->setObjectName(QStringLiteral("L_Restitution"));

        gridLayout_2->addWidget(L_Restitution, 0, 0, 1, 1);

        LE_Restitution = new QLineEdit(widget);
        LE_Restitution->setObjectName(QStringLiteral("LE_Restitution"));

        gridLayout_2->addWidget(LE_Restitution, 0, 1, 1, 1);

        L_StiffnessRatio = new QLabel(widget);
        L_StiffnessRatio->setObjectName(QStringLiteral("L_StiffnessRatio"));

        gridLayout_2->addWidget(L_StiffnessRatio, 1, 0, 1, 1);

        LE_StiffnessRatio = new QLineEdit(widget);
        LE_StiffnessRatio->setObjectName(QStringLiteral("LE_StiffnessRatio"));

        gridLayout_2->addWidget(LE_StiffnessRatio, 1, 1, 1, 1);

        L_Friction = new QLabel(widget);
        L_Friction->setObjectName(QStringLiteral("L_Friction"));

        gridLayout_2->addWidget(L_Friction, 2, 0, 1, 1);

        LE_Friction = new QLineEdit(widget);
        LE_Friction->setObjectName(QStringLiteral("LE_Friction"));

        gridLayout_2->addWidget(LE_Friction, 2, 1, 1, 1);

        L_Cohesion = new QLabel(widget);
        L_Cohesion->setObjectName(QStringLiteral("L_Cohesion"));

        gridLayout_2->addWidget(L_Cohesion, 3, 0, 1, 1);

        LE_Cohesion = new QLineEdit(widget);
        LE_Cohesion->setObjectName(QStringLiteral("LE_Cohesion"));

        gridLayout_2->addWidget(LE_Cohesion, 3, 1, 1, 1);

        TB_METHOD->addTab(T_DHS, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        TB_METHOD->addTab(tab_2, QString());
        layoutWidget1 = new QWidget(DLG_ContactPair);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(150, 350, 181, 25));
        horizontalLayout = new QHBoxLayout(layoutWidget1);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(layoutWidget1);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(layoutWidget1);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout->addWidget(PB_Cancle);

        layoutWidget2 = new QWidget(DLG_ContactPair);
        layoutWidget2->setObjectName(QStringLiteral("layoutWidget2"));
        layoutWidget2->setGeometry(QRect(10, 20, 321, 22));
        horizontalLayout_2 = new QHBoxLayout(layoutWidget2);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Name = new QLabel(layoutWidget2);
        L_Name->setObjectName(QStringLiteral("L_Name"));

        horizontalLayout_2->addWidget(L_Name);

        LE_Name = new QLineEdit(layoutWidget2);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));

        horizontalLayout_2->addWidget(LE_Name);

        CHB_IgnoreCondition = new QCheckBox(DLG_ContactPair);
        CHB_IgnoreCondition->setObjectName(QStringLiteral("CHB_IgnoreCondition"));
        CHB_IgnoreCondition->setGeometry(QRect(20, 324, 121, 16));
        layoutWidget3 = new QWidget(DLG_ContactPair);
        layoutWidget3->setObjectName(QStringLiteral("layoutWidget3"));
        layoutWidget3->setGeometry(QRect(136, 320, 201, 22));
        horizontalLayout_3 = new QHBoxLayout(layoutWidget3);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        L_IgnoreTime = new QLabel(layoutWidget3);
        L_IgnoreTime->setObjectName(QStringLiteral("L_IgnoreTime"));

        horizontalLayout_3->addWidget(L_IgnoreTime);

        LE_IgnoreTime = new QLineEdit(layoutWidget3);
        LE_IgnoreTime->setObjectName(QStringLiteral("LE_IgnoreTime"));

        horizontalLayout_3->addWidget(LE_IgnoreTime);

        QWidget::setTabOrder(LE_Name, TB_METHOD);
        QWidget::setTabOrder(TB_METHOD, CB_FirstObject);
        QWidget::setTabOrder(CB_FirstObject, CB_SecondObject);
        QWidget::setTabOrder(CB_SecondObject, LE_Restitution);
        QWidget::setTabOrder(LE_Restitution, LE_StiffnessRatio);
        QWidget::setTabOrder(LE_StiffnessRatio, LE_Friction);
        QWidget::setTabOrder(LE_Friction, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);

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
        L_Cohesion->setText(QApplication::translate("DLG_ContactPair", "Cohesion", Q_NULLPTR));
        LE_Cohesion->setText(QString());
        TB_METHOD->setTabText(TB_METHOD->indexOf(T_DHS), QApplication::translate("DLG_ContactPair", "DHS", Q_NULLPTR));
        TB_METHOD->setTabText(TB_METHOD->indexOf(tab_2), QApplication::translate("DLG_ContactPair", "Tab 2", Q_NULLPTR));
        PB_Ok->setText(QApplication::translate("DLG_ContactPair", "Ok", Q_NULLPTR));
        PB_Cancle->setText(QApplication::translate("DLG_ContactPair", "Cancle", Q_NULLPTR));
        L_Name->setText(QApplication::translate("DLG_ContactPair", "Name", Q_NULLPTR));
        CHB_IgnoreCondition->setText(QApplication::translate("DLG_ContactPair", "Ignore condition", Q_NULLPTR));
        L_IgnoreTime->setText(QApplication::translate("DLG_ContactPair", "Time", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_ContactPair: public Ui_DLG_ContactPair {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONTACTPAIR_H
