/********************************************************************************
** Form generated from reading UI file 'GenParticle.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GENPARTICLE_H
#define UI_GENPARTICLE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_Particle
{
public:
    QGroupBox *GB_ParticleData;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout_3;
    QLabel *L_BaseGeometry;
    QLabel *L_Radius;
    QVBoxLayout *verticalLayout_4;
    QComboBox *CB_BaseGeometry;
    QLineEdit *LE_Radius;
    QGroupBox *GB_Function;
    QCheckBox *CHB_StackParticle;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_4;
    QVBoxLayout *verticalLayout_5;
    QLabel *L_StackNumber;
    QLabel *L_StackTimeInterval;
    QVBoxLayout *verticalLayout_6;
    QLineEdit *LE_StackNumber;
    QLineEdit *LE_StackTimeInterval;
    QGroupBox *GB_Information;
    QWidget *widget2;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout;
    QLabel *L_NumParticle;
    QLabel *L_TotalMass;
    QLabel *L_Spacing;
    QLabel *L_Size;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *LE_NumParticle;
    QLineEdit *LE_TotalMass;
    QLineEdit *LE_Spacing;
    QLineEdit *LE_Size;
    QWidget *widget3;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;

    void setupUi(QDialog *DLG_Particle)
    {
        if (DLG_Particle->objectName().isEmpty())
            DLG_Particle->setObjectName(QStringLiteral("DLG_Particle"));
        DLG_Particle->resize(391, 425);
        DLG_Particle->setModal(true);
        GB_ParticleData = new QGroupBox(DLG_Particle);
        GB_ParticleData->setObjectName(QStringLiteral("GB_ParticleData"));
        GB_ParticleData->setGeometry(QRect(10, 10, 371, 101));
        widget = new QWidget(GB_ParticleData);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(22, 21, 331, 61));
        horizontalLayout_2 = new QHBoxLayout(widget);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        L_BaseGeometry = new QLabel(widget);
        L_BaseGeometry->setObjectName(QStringLiteral("L_BaseGeometry"));

        verticalLayout_3->addWidget(L_BaseGeometry);

        L_Radius = new QLabel(widget);
        L_Radius->setObjectName(QStringLiteral("L_Radius"));

        verticalLayout_3->addWidget(L_Radius);


        horizontalLayout_2->addLayout(verticalLayout_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        CB_BaseGeometry = new QComboBox(widget);
        CB_BaseGeometry->setObjectName(QStringLiteral("CB_BaseGeometry"));

        verticalLayout_4->addWidget(CB_BaseGeometry);

        LE_Radius = new QLineEdit(widget);
        LE_Radius->setObjectName(QStringLiteral("LE_Radius"));

        verticalLayout_4->addWidget(LE_Radius);


        horizontalLayout_2->addLayout(verticalLayout_4);

        GB_Function = new QGroupBox(DLG_Particle);
        GB_Function->setObjectName(QStringLiteral("GB_Function"));
        GB_Function->setGeometry(QRect(10, 122, 371, 101));
        CHB_StackParticle = new QCheckBox(GB_Function);
        CHB_StackParticle->setObjectName(QStringLiteral("CHB_StackParticle"));
        CHB_StackParticle->setGeometry(QRect(20, 20, 101, 16));
        widget1 = new QWidget(GB_Function);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(20, 41, 331, 50));
        horizontalLayout_4 = new QHBoxLayout(widget1);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        L_StackNumber = new QLabel(widget1);
        L_StackNumber->setObjectName(QStringLiteral("L_StackNumber"));

        verticalLayout_5->addWidget(L_StackNumber);

        L_StackTimeInterval = new QLabel(widget1);
        L_StackTimeInterval->setObjectName(QStringLiteral("L_StackTimeInterval"));

        verticalLayout_5->addWidget(L_StackTimeInterval);


        horizontalLayout_4->addLayout(verticalLayout_5);

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        LE_StackNumber = new QLineEdit(widget1);
        LE_StackNumber->setObjectName(QStringLiteral("LE_StackNumber"));

        verticalLayout_6->addWidget(LE_StackNumber);

        LE_StackTimeInterval = new QLineEdit(widget1);
        LE_StackTimeInterval->setObjectName(QStringLiteral("LE_StackTimeInterval"));

        verticalLayout_6->addWidget(LE_StackTimeInterval);


        horizontalLayout_4->addLayout(verticalLayout_6);

        GB_Information = new QGroupBox(DLG_Particle);
        GB_Information->setObjectName(QStringLiteral("GB_Information"));
        GB_Information->setGeometry(QRect(10, 234, 371, 151));
        widget2 = new QWidget(GB_Information);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(26, 24, 331, 111));
        horizontalLayout = new QHBoxLayout(widget2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 6, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        L_NumParticle = new QLabel(widget2);
        L_NumParticle->setObjectName(QStringLiteral("L_NumParticle"));

        verticalLayout->addWidget(L_NumParticle);

        L_TotalMass = new QLabel(widget2);
        L_TotalMass->setObjectName(QStringLiteral("L_TotalMass"));

        verticalLayout->addWidget(L_TotalMass);

        L_Spacing = new QLabel(widget2);
        L_Spacing->setObjectName(QStringLiteral("L_Spacing"));

        verticalLayout->addWidget(L_Spacing);

        L_Size = new QLabel(widget2);
        L_Size->setObjectName(QStringLiteral("L_Size"));

        verticalLayout->addWidget(L_Size);


        horizontalLayout->addLayout(verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        LE_NumParticle = new QLineEdit(widget2);
        LE_NumParticle->setObjectName(QStringLiteral("LE_NumParticle"));
        LE_NumParticle->setReadOnly(true);

        verticalLayout_2->addWidget(LE_NumParticle);

        LE_TotalMass = new QLineEdit(widget2);
        LE_TotalMass->setObjectName(QStringLiteral("LE_TotalMass"));
        LE_TotalMass->setReadOnly(true);

        verticalLayout_2->addWidget(LE_TotalMass);

        LE_Spacing = new QLineEdit(widget2);
        LE_Spacing->setObjectName(QStringLiteral("LE_Spacing"));
        LE_Spacing->setReadOnly(true);

        verticalLayout_2->addWidget(LE_Spacing);

        LE_Size = new QLineEdit(widget2);
        LE_Size->setObjectName(QStringLiteral("LE_Size"));
        LE_Size->setReadOnly(true);

        verticalLayout_2->addWidget(LE_Size);


        horizontalLayout->addLayout(verticalLayout_2);

        widget3 = new QWidget(DLG_Particle);
        widget3->setObjectName(QStringLiteral("widget3"));
        widget3->setGeometry(QRect(220, 390, 158, 25));
        horizontalLayout_3 = new QHBoxLayout(widget3);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(widget3);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout_3->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(widget3);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout_3->addWidget(PB_Cancle);

        QWidget::setTabOrder(CB_BaseGeometry, LE_Radius);
        QWidget::setTabOrder(LE_Radius, CHB_StackParticle);
        QWidget::setTabOrder(CHB_StackParticle, LE_StackNumber);
        QWidget::setTabOrder(LE_StackNumber, LE_StackTimeInterval);
        QWidget::setTabOrder(LE_StackTimeInterval, LE_NumParticle);
        QWidget::setTabOrder(LE_NumParticle, LE_TotalMass);
        QWidget::setTabOrder(LE_TotalMass, LE_Spacing);
        QWidget::setTabOrder(LE_Spacing, LE_Size);
        QWidget::setTabOrder(LE_Size, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);

        retranslateUi(DLG_Particle);

        QMetaObject::connectSlotsByName(DLG_Particle);
    } // setupUi

    void retranslateUi(QDialog *DLG_Particle)
    {
        DLG_Particle->setWindowTitle(QApplication::translate("DLG_Particle", "Particle Creating Dialog", 0));
        GB_ParticleData->setTitle(QApplication::translate("DLG_Particle", "Particle data", 0));
        L_BaseGeometry->setText(QApplication::translate("DLG_Particle", "Base geometry", 0));
        L_Radius->setText(QApplication::translate("DLG_Particle", "Radius", 0));
        GB_Function->setTitle(QApplication::translate("DLG_Particle", "Function", 0));
        CHB_StackParticle->setText(QApplication::translate("DLG_Particle", "Stack particle", 0));
        L_StackNumber->setText(QApplication::translate("DLG_Particle", "Number", 0));
        L_StackTimeInterval->setText(QApplication::translate("DLG_Particle", "Time interval", 0));
        GB_Information->setTitle(QApplication::translate("DLG_Particle", "Information", 0));
        L_NumParticle->setText(QApplication::translate("DLG_Particle", "Num. particle", 0));
        L_TotalMass->setText(QApplication::translate("DLG_Particle", "Total mass", 0));
        L_Spacing->setText(QApplication::translate("DLG_Particle", "Spacing", 0));
        L_Size->setText(QApplication::translate("DLG_Particle", "Size", 0));
        PB_Ok->setText(QApplication::translate("DLG_Particle", "Ok", 0));
        PB_Cancle->setText(QApplication::translate("DLG_Particle", "Cancle", 0));
    } // retranslateUi

};

namespace Ui {
    class DLG_Particle: public Ui_DLG_Particle {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GENPARTICLE_H
