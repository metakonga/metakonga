#include "resultTable.h"
#include "MyTableWidgetItem.h"
#include <QtWidgets>

resultTable::resultTable(QWidget *parent) : QWidget(parent)
{
	rtable = new QTableWidget(0, 18);
	rtable->setSelectionBehavior(QAbstractItemView::SelectRows);

	QStringList labels;
	labels << kor("아이디") << kor("유효권수") << kor("중심경(mm)") << kor("재질경(mm)") << kor("자유장(mm)") << kor("강성(kgf/mm)") << kor("질량(kg)") << kor("밀착고(mm)") << kor("하중(kgf)") << kor("밀착고하중(kgf)") << kor("최대응력(kgf/mm^2)") << kor("밀착고응력(kgf/mm^2)") << kor("유효포텐셜(N*m)") << kor("효율(E)") << kor("전달에너지(N*m)") << kor("끝단속도(m/s)") << kor("스프링지수") << kor("종횡비");
	rtable->setHorizontalHeaderLabels(labels);
	rtable->verticalHeader()->hide();
	rtable->setShowGrid(true);
	//rtable->setSortingEnabled(true);

	QGridLayout *mainLayout = new QGridLayout;
	mainLayout->setSizeConstraint(QLayout::SetNoConstraint);
	mainLayout->addWidget(rtable, 0, 0);
	setLayout(mainLayout);
	setWindowTitle(tr("Result table"));
	resize(700, 300);
	setWindowModality(Qt::WindowModality::ApplicationModal);

	//connect(rtable, SIGNAL(clicked(QModelIndex)), this, SLOT(actionClick(QModelIndex)));
}

resultTable::~resultTable()
{
	if (rtable) delete rtable; rtable = NULL;
}

void resultTable::reset()
{
	rtable->setSortingEnabled(false);
	rtable->clearContents();
	rtable->setRowCount(0);
}

void resultTable::actionClick(QModelIndex index)
{
	int row = index.row();
	int col = index.column();
	
}

void resultTable::setTable(resultSet &ref_result, algebra::vector<resultSet> &results)
{
	rtable->setSortingEnabled(false);
	rtable->clearContents();
	rtable->setRowCount(0);

	unsigned int row = 0;
	unsigned int i = 0;
	QString str;
	QTextStream(&str) << i;  MyTableWidgetItem *ID = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.N;  MyTableWidgetItem *N = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.D;  MyTableWidgetItem *D = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.d;  MyTableWidgetItem *d = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.FreeLength;  MyTableWidgetItem *FreeLength = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.k;  MyTableWidgetItem *k = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.Mass;  MyTableWidgetItem *Mass = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.B_Height;  MyTableWidgetItem *B_Height = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.P;  MyTableWidgetItem *P = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.P_BH;  MyTableWidgetItem *P_BH = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.Sc;  MyTableWidgetItem *Sc = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.Sc_BH;  MyTableWidgetItem *Sc_BH = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.PE_act;  MyTableWidgetItem *PE_act = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.Efficiency;  MyTableWidgetItem *Efficiency = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.transferEnergy;  MyTableWidgetItem *transferEnergy = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.v_ep;  MyTableWidgetItem *v_ep = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.C;  MyTableWidgetItem *C = new MyTableWidgetItem(str); str.clear();
	QTextStream(&str) << ref_result.AR;  MyTableWidgetItem *AR = new MyTableWidgetItem(str); str.clear();

	rtable->insertRow(row);
	rtable->setItem(row, 0, ID);
	rtable->setItem(row, 1, N);
	rtable->setItem(row, 2, D);
	rtable->setItem(row, 3, d);
	rtable->setItem(row, 4, FreeLength);
	rtable->setItem(row, 5, k);
	rtable->setItem(row, 6, Mass);
	rtable->setItem(row, 7, B_Height);
	rtable->setItem(row, 8, P);
	rtable->setItem(row, 9, P_BH);
	rtable->setItem(row, 10, Sc);
	rtable->setItem(row, 11, Sc_BH);
	rtable->setItem(row, 12, PE_act);
	rtable->setItem(row, 13, Efficiency);
	rtable->setItem(row, 14, transferEnergy);
	rtable->setItem(row, 15, v_ep);
	rtable->setItem(row, 16, C);
	rtable->setItem(row, 17, AR);
	row++;
	for (unsigned int i = 0; i < results.sizes(); i++){
		resultSet result = results(i);
		
		QTextStream(&str) << i+1;  MyTableWidgetItem *ID = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.N;  MyTableWidgetItem *N = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.D;  MyTableWidgetItem *D = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.d;  MyTableWidgetItem *d = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.FreeLength;  MyTableWidgetItem *FreeLength = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.k;  MyTableWidgetItem *k = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.Mass;  MyTableWidgetItem *Mass = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.B_Height;  MyTableWidgetItem *B_Height = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.P;  MyTableWidgetItem *P = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.P_BH;  MyTableWidgetItem *P_BH = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.Sc;  MyTableWidgetItem *Sc = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.Sc_BH;  MyTableWidgetItem *Sc_BH = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.PE_act;  MyTableWidgetItem *PE_act = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.Efficiency;  MyTableWidgetItem *Efficiency = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.transferEnergy;  MyTableWidgetItem *transferEnergy = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.v_ep;  MyTableWidgetItem *v_ep = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.C;  MyTableWidgetItem *C = new MyTableWidgetItem(str); str.clear();
		QTextStream(&str) << result.AR;  MyTableWidgetItem *AR = new MyTableWidgetItem(str); str.clear();

		rtable->insertRow(row);
		rtable->setItem(row, 0, ID);
		rtable->setItem(row, 1, N);
		rtable->setItem(row, 2, D);
		rtable->setItem(row, 3, d);
		rtable->setItem(row, 4, FreeLength);
		rtable->setItem(row, 5, k);
		rtable->setItem(row, 6, Mass);
		rtable->setItem(row, 7, B_Height);
		rtable->setItem(row, 8, P);
		rtable->setItem(row, 9, P_BH);
		rtable->setItem(row, 10, Sc);
		rtable->setItem(row, 11, Sc_BH);
		rtable->setItem(row, 12, PE_act);
		rtable->setItem(row, 13, Efficiency);
		rtable->setItem(row, 14, transferEnergy);
		rtable->setItem(row, 15, v_ep);
		rtable->setItem(row, 16, C);
		rtable->setItem(row, 17, AR);
		row++;
	}
	//rtable->sortByColumn(0, Qt::SortOrder::AscendingOrder);
	rtable->setSortingEnabled(true);
}