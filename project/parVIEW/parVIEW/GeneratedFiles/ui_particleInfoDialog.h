/********************************************************************************
** Form generated from reading UI file 'particleInfoDialogfl4712.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef PARTICLEINFODIALOGFL4712_H
#define PARTICLEINFODIALOGFL4712_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_PInfoDialog
{
public:
    QPushButton *BChange;
    QWidget *widget;
    QGridLayout *gridLayout;
    QLabel *LPid;
    QScrollBar *PHSlider;
    QLabel *LSelectedPid;
    QLineEdit *LEPid;
    QLabel *LTotalParticle;
    QLineEdit *LETotalParticle;
    QWidget *widget1;
    QGridLayout *gridLayout_2;
    QLabel *LPosition;
    QLineEdit *LEPosition;
    QLabel *LVelocity;
    QLineEdit *LEVelocity;
    QWidget *widget2;
    QGridLayout *gridLayout_3;
    QLabel *LFressure;
    QLineEdit *LEPressure;
    QLabel *LFreeSurfaceValue;
    QLineEdit *LEFreeSurfaceValue;

    void setupUi(QDialog *PInfoDialog)
    {
        if (PInfoDialog->objectName().isEmpty())
            PInfoDialog->setObjectName(QStringLiteral("PInfoDialog"));
        PInfoDialog->resize(616, 146);
        BChange = new QPushButton(PInfoDialog);
        BChange->setObjectName(QStringLiteral("BChange"));
        BChange->setGeometry(QRect(540, 20, 71, 111));
        BChange->setAutoDefault(false);
        widget = new QWidget(PInfoDialog);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 20, 521, 45));
        gridLayout = new QGridLayout(widget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        LPid = new QLabel(widget);
        LPid->setObjectName(QStringLiteral("LPid"));

        gridLayout->addWidget(LPid, 0, 0, 1, 1);

        PHSlider = new QScrollBar(widget);
        PHSlider->setObjectName(QStringLiteral("PHSlider"));
        PHSlider->setToolTipDuration(-5);
        PHSlider->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(PHSlider, 0, 1, 1, 3);

        LSelectedPid = new QLabel(widget);
        LSelectedPid->setObjectName(QStringLiteral("LSelectedPid"));

        gridLayout->addWidget(LSelectedPid, 1, 0, 1, 1);

        LEPid = new QLineEdit(widget);
        LEPid->setObjectName(QStringLiteral("LEPid"));

        gridLayout->addWidget(LEPid, 1, 1, 1, 1);

        LTotalParticle = new QLabel(widget);
        LTotalParticle->setObjectName(QStringLiteral("LTotalParticle"));

        gridLayout->addWidget(LTotalParticle, 1, 2, 1, 1);

        LETotalParticle = new QLineEdit(widget);
        LETotalParticle->setObjectName(QStringLiteral("LETotalParticle"));

        gridLayout->addWidget(LETotalParticle, 1, 3, 1, 1);

        widget1 = new QWidget(PInfoDialog);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(10, 80, 261, 48));
        gridLayout_2 = new QGridLayout(widget1);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        LPosition = new QLabel(widget1);
        LPosition->setObjectName(QStringLiteral("LPosition"));

        gridLayout_2->addWidget(LPosition, 0, 0, 1, 1);

        LEPosition = new QLineEdit(widget1);
        LEPosition->setObjectName(QStringLiteral("LEPosition"));

        gridLayout_2->addWidget(LEPosition, 0, 1, 1, 1);

        LVelocity = new QLabel(widget1);
        LVelocity->setObjectName(QStringLiteral("LVelocity"));

        gridLayout_2->addWidget(LVelocity, 1, 0, 1, 1);

        LEVelocity = new QLineEdit(widget1);
        LEVelocity->setObjectName(QStringLiteral("LEVelocity"));

        gridLayout_2->addWidget(LEVelocity, 1, 1, 1, 1);

        widget2 = new QWidget(PInfoDialog);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(280, 80, 251, 48));
        gridLayout_3 = new QGridLayout(widget2);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        LFressure = new QLabel(widget2);
        LFressure->setObjectName(QStringLiteral("LFressure"));

        gridLayout_3->addWidget(LFressure, 0, 0, 1, 1);

        LEPressure = new QLineEdit(widget2);
        LEPressure->setObjectName(QStringLiteral("LEPressure"));

        gridLayout_3->addWidget(LEPressure, 0, 1, 1, 1);

        LFreeSurfaceValue = new QLabel(widget2);
        LFreeSurfaceValue->setObjectName(QStringLiteral("LFreeSurfaceValue"));

        gridLayout_3->addWidget(LFreeSurfaceValue, 1, 0, 1, 1);

        LEFreeSurfaceValue = new QLineEdit(widget2);
        LEFreeSurfaceValue->setObjectName(QStringLiteral("LEFreeSurfaceValue"));

        gridLayout_3->addWidget(LEFreeSurfaceValue, 1, 1, 1, 1);


        retranslateUi(PInfoDialog);

        QMetaObject::connectSlotsByName(PInfoDialog);
    } // setupUi

    void retranslateUi(QDialog *PInfoDialog)
    {
        PInfoDialog->setWindowTitle(QApplication::translate("PInfoDialog", "Dialog", 0));
        BChange->setText(QApplication::translate("PInfoDialog", "C.SR", 0));
        LPid->setText(QApplication::translate("PInfoDialog", "P.ID", 0));
        LSelectedPid->setText(QApplication::translate("PInfoDialog", "Selected P.ID", 0));
        LTotalParticle->setText(QApplication::translate("PInfoDialog", "Total Particle", 0));
        LPosition->setText(QApplication::translate("PInfoDialog", "Position", 0));
        LVelocity->setText(QApplication::translate("PInfoDialog", "Velocity", 0));
        LFressure->setText(QApplication::translate("PInfoDialog", "Pressure", 0));
        LFreeSurfaceValue->setText(QApplication::translate("PInfoDialog", "FreeSurface Value", 0));
    } // retranslateUi

};

namespace Ui {
    class PInfoDialog: public Ui_PInfoDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // PARTICLEINFODIALOGFL4712_H
