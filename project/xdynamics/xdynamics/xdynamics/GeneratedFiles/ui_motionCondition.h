/********************************************************************************
** Form generated from reading UI file 'motionCondition.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MOTIONCONDITION_H
#define UI_MOTIONCONDITION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_MotionCondition
{
public:
    QGroupBox *GB_MotionCondition;
    QRadioButton *RB_ConstantVelocity;
    QRadioButton *RB_NoCondition;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout;
    QLabel *L_StartTime;
    QLabel *L_EndTime;
    QLabel *L_Value;
    QLabel *L_Direction;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *LE_StartTime;
    QLineEdit *LE_EndTime;
    QLineEdit *LE_Value;
    QLineEdit *LE_Direction;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout;
    QLabel *L_Name;
    QLineEdit *LE_Name;
    QWidget *widget2;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;

    void setupUi(QDialog *DLG_MotionCondition)
    {
        if (DLG_MotionCondition->objectName().isEmpty())
            DLG_MotionCondition->setObjectName(QStringLiteral("DLG_MotionCondition"));
        DLG_MotionCondition->resize(272, 239);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(DLG_MotionCondition->sizePolicy().hasHeightForWidth());
        DLG_MotionCondition->setSizePolicy(sizePolicy);
        GB_MotionCondition = new QGroupBox(DLG_MotionCondition);
        GB_MotionCondition->setObjectName(QStringLiteral("GB_MotionCondition"));
        GB_MotionCondition->setGeometry(QRect(10, 40, 251, 151));
        RB_ConstantVelocity = new QRadioButton(GB_MotionCondition);
        RB_ConstantVelocity->setObjectName(QStringLiteral("RB_ConstantVelocity"));
        RB_ConstantVelocity->setGeometry(QRect(120, 20, 121, 16));
        RB_NoCondition = new QRadioButton(GB_MotionCondition);
        RB_NoCondition->setObjectName(QStringLiteral("RB_NoCondition"));
        RB_NoCondition->setGeometry(QRect(10, 20, 90, 16));
        RB_NoCondition->setChecked(true);
        widget = new QWidget(GB_MotionCondition);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 42, 231, 102));
        horizontalLayout_2 = new QHBoxLayout(widget);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        L_StartTime = new QLabel(widget);
        L_StartTime->setObjectName(QStringLiteral("L_StartTime"));

        verticalLayout->addWidget(L_StartTime);

        L_EndTime = new QLabel(widget);
        L_EndTime->setObjectName(QStringLiteral("L_EndTime"));

        verticalLayout->addWidget(L_EndTime);

        L_Value = new QLabel(widget);
        L_Value->setObjectName(QStringLiteral("L_Value"));

        verticalLayout->addWidget(L_Value);

        L_Direction = new QLabel(widget);
        L_Direction->setObjectName(QStringLiteral("L_Direction"));

        verticalLayout->addWidget(L_Direction);


        horizontalLayout_2->addLayout(verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        LE_StartTime = new QLineEdit(widget);
        LE_StartTime->setObjectName(QStringLiteral("LE_StartTime"));

        verticalLayout_2->addWidget(LE_StartTime);

        LE_EndTime = new QLineEdit(widget);
        LE_EndTime->setObjectName(QStringLiteral("LE_EndTime"));

        verticalLayout_2->addWidget(LE_EndTime);

        LE_Value = new QLineEdit(widget);
        LE_Value->setObjectName(QStringLiteral("LE_Value"));

        verticalLayout_2->addWidget(LE_Value);

        LE_Direction = new QLineEdit(widget);
        LE_Direction->setObjectName(QStringLiteral("LE_Direction"));

        verticalLayout_2->addWidget(LE_Direction);


        horizontalLayout_2->addLayout(verticalLayout_2);

        widget1 = new QWidget(DLG_MotionCondition);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(10, 10, 251, 22));
        horizontalLayout = new QHBoxLayout(widget1);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        L_Name = new QLabel(widget1);
        L_Name->setObjectName(QStringLiteral("L_Name"));

        horizontalLayout->addWidget(L_Name);

        LE_Name = new QLineEdit(widget1);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));
        LE_Name->setReadOnly(true);

        horizontalLayout->addWidget(LE_Name);

        widget2 = new QWidget(DLG_MotionCondition);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(100, 200, 158, 31));
        horizontalLayout_3 = new QHBoxLayout(widget2);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(widget2);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout_3->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(widget2);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout_3->addWidget(PB_Cancle);

        QWidget::setTabOrder(LE_Name, RB_NoCondition);
        QWidget::setTabOrder(RB_NoCondition, RB_ConstantVelocity);
        QWidget::setTabOrder(RB_ConstantVelocity, LE_StartTime);
        QWidget::setTabOrder(LE_StartTime, LE_EndTime);
        QWidget::setTabOrder(LE_EndTime, LE_Value);
        QWidget::setTabOrder(LE_Value, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);

        retranslateUi(DLG_MotionCondition);

        QMetaObject::connectSlotsByName(DLG_MotionCondition);
    } // setupUi

    void retranslateUi(QDialog *DLG_MotionCondition)
    {
        DLG_MotionCondition->setWindowTitle(QApplication::translate("DLG_MotionCondition", "Dialog", nullptr));
        GB_MotionCondition->setTitle(QApplication::translate("DLG_MotionCondition", "Motion condition", nullptr));
        RB_ConstantVelocity->setText(QApplication::translate("DLG_MotionCondition", "Constant velocity", nullptr));
        RB_NoCondition->setText(QApplication::translate("DLG_MotionCondition", "No condition", nullptr));
        L_StartTime->setText(QApplication::translate("DLG_MotionCondition", "Start time", nullptr));
        L_EndTime->setText(QApplication::translate("DLG_MotionCondition", "End time", nullptr));
        L_Value->setText(QApplication::translate("DLG_MotionCondition", "Value", nullptr));
        L_Direction->setText(QApplication::translate("DLG_MotionCondition", "Direction", nullptr));
        L_Name->setText(QApplication::translate("DLG_MotionCondition", "Name", nullptr));
        PB_Ok->setText(QApplication::translate("DLG_MotionCondition", "Ok", nullptr));
        PB_Cancle->setText(QApplication::translate("DLG_MotionCondition", "Cancle", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_MotionCondition: public Ui_DLG_MotionCondition {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MOTIONCONDITION_H
