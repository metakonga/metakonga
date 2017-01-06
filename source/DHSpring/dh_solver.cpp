#include "dh_solver.h"
#include <iostream>
#include <cmath>
#include "algebra.h"
#include <QFile>
#include <QMessagebox>

#include <lapack/f2c.h>
#include <lapack/clapack.h>

using namespace algebra;

dh_solver::dh_solver()

{
	reset();
}

dh_solver::~dh_solver()
{

}

void dh_solver::reset()
{
	 N = 0.0;
	 D = 0.0;
	 d = 0.0;
	 FreeLength = 0.0;
	 InstallLength = 0.0;
	 Stroke = 0.0;
	 DisBH = 0.0;

	 NLower = 0.0;
	 NUpper = 0.0;
	 DLower = 0.0;
	 DUpper = 0.0;
	 dLower = 0.0;
	 dUpper = 0.0;
	 FreeLengthLower = 0.0;
	 FreeLengthUpper = 0.0;
	 deltaN = 0.0;
	 deltaD = 0.0;
	 deltad = 0.0;
	 deltaFreeLength = 0.0;
	 deltaInstLength = 0.0;

	 density = 0.0;
	 eq_mass = 0.0;

	 PreComp = 0.0;
	 TotalComp = 0.0;
	 BHLimit = 0.0;
	 shearModulus = 0.0;

	 cMaxStress = 0.0;
	 cMinStress = 0.0;
	 cMaxStiffness = 0.0;
	 cMinStiffness = 0.0;
	 cBHStress = 0.0;
	 cMass = 0.0;
	 cMaxSpringIndex = 0.0;
	 cMinSpringIndex = 0.0;
	 cAspectRatio = 0.0;
	 cWeight = 0.0;
	 cPotential = 0.0;

	 results.dealloc();
}

void dh_solver::setTestExample()
{
	N = 8;
	D = 130;
	d = 31;
	FreeLength = 450;
	InstallLength = 430;
	Stroke = 75;
	DisBH = 0.8;

	NLower = -3;
	NUpper = 3;
	DLower = -30;
	DUpper = 30;
	dLower = -5;
	dUpper = 5;
	FreeLengthLower = -60;
	FreeLengthUpper = 20;
	deltaN = 0.1;
	deltaD = 1;
	deltad = 1;
	deltaFreeLength = 1;
	deltaInstLength = 1;

	density = 7850e-9;
	eq_mass = 16.22;

	PreComp = FreeLength - InstallLength;
	TotalComp = PreComp + Stroke;
	BHLimit = FreeLength - (TotalComp / DisBH);
	shearModulus = 8000;
	dt = 0.00001;
	et = 0.5;
}

double dh_solver::lumped(double k_refer, double Mass, double Mass_EB, double Mass_eq_linkage, double tComp, double pComp)
{
	double v_ep = 0;
	double x_i_end = 0;
	double xd_i_end = 0;
	double k_ref = k_refer * 9.81 * 1000;
	double Mass_segment = (Mass - (2 * Mass_EB)) / 9;
	double k18 = k_ref * 18;
	double k9 = k_ref * 9;
	double c = 300;
	double p = k_ref * tComp * 0.001;
	matrix<double> M(10, 10, 0.0);
	M(0, 0) = Mass_segment; M(1, 1) = Mass_segment; M(2, 2) = Mass_segment; M(3, 3) = Mass_segment; M(4, 4) = Mass_segment;
	M(5, 5) = Mass_segment; M(6, 6) = Mass_segment; M(7, 7) = Mass_segment; M(8, 8) = Mass_segment; M(9, 9) = Mass_EB + Mass_eq_linkage;
	matrix<double> K(10, 10, 0.0);
	K(0, 0) = K(8, 8) = k18 + k9; 
	K(0, 1) = K(1, 0) = -k9; 
	K(1, 2) = K(2, 1) = -k9;
	K(2, 3) = K(3, 2) = -k9;
	K(3, 4) = K(4, 3) = -k9;
	K(4, 5) = K(5, 4) = -k9;
	K(5, 6) = K(6, 5) = -k9;
	K(6, 7) = K(7, 6) = -k9;
	K(7, 8) = K(8, 7) = -k9;
	K(8, 9) = K(9, 8) = -k18;
	K(1, 1) = K(2, 2) = K(3, 3) = K(4, 4) = K(5, 5) = K(6, 6) = K(7, 7) = 2 * k9;
	K(9, 9) = k18;

	matrix<double> C(10, 10, 0.0);
	C(0, 0) = C(8, 8) = c + c;
	C(0, 1) = C(1, 0) = -c;
	C(1, 2) = C(2, 1) = -c;
	C(2, 3) = C(3, 2) = -c;
	C(3, 4) = C(4, 3) = -c;
	C(4, 5) = C(5, 4) = -c;
	C(5, 6) = C(6, 5) = -c;
	C(6, 7) = C(7, 6) = -c;
	C(7, 8) = C(8, 7) = -c;
	C(8, 9) = C(9, 8) = -c;
	C(1, 1) = C(2, 2) = C(3, 3) = C(4, 4) = C(5, 5) = C(6, 6) = C(7, 7) = 2 * c;
	C(9, 9) = c;

	algebra::vector<double> d(10, 0.0);
	d(0) = d(9) = -p / k18;
	d(1) = d(2) = d(3) = d(4) = d(5) = d(6) = d(7) = d(8) = -p / k9;
	algebra::vector<double> x(10, 0.0);
	for (unsigned int  i = 0; i < 10; i++)
		for (unsigned int j = 0; j <= i; j++)
			x(i) += d(j);
	algebra::vector<double> xd(10, 0.0);
	algebra::vector<double> xdd(10, 0.0);
	algebra::vector<double> tmp0(10, 0.0);
	algebra::vector<double> tmp1(10, 0.0);

	char trans;
	integer lapack_one = 1;
	doublereal lapack_zero = 0;
	doublereal lapack_mone = -1;
	integer lapack_info = 0;
	integer ptDof = 10;
	integer *permutation = new integer[10];
	for (double time = 0; time <= et + 1e-10; time += dt){
		trans = 'n';
		matrix<double> _M(M);
		matrix<double> _K(K);
		matrix<double> _C(C);
		algebra::vector<double> _x(x);
		algebra::vector<double> _xd(xd);
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _K.getDataPointer(), &ptDof, x.get_ptr(), &lapack_one, &lapack_zero, tmp0.get_ptr(), &lapack_one);
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _C.getDataPointer(), &ptDof, xd.get_ptr(), &lapack_one, &lapack_zero, tmp1.get_ptr(), &lapack_one);
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp0.get_ptr(), &ptDof, &lapack_info);
		_M = M;
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp1.get_ptr(), &ptDof, &lapack_info);

		algebra::vector<double> K1(tmp0 + tmp1);

		_x = x +  xd * (0.5 * dt) + K1 * (0.125 * dt * dt);
		_xd = xd + K1 * (0.5 * dt); _M = M; _K = K; _C = C;
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _K.getDataPointer(), &ptDof, _x.get_ptr(), &lapack_one, &lapack_zero, tmp0.get_ptr(), &lapack_one);
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _C.getDataPointer(), &ptDof, _xd.get_ptr(), &lapack_one, &lapack_zero, tmp1.get_ptr(), &lapack_one);
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp0.get_ptr(), &ptDof, &lapack_info); _M = M;
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp1.get_ptr(), &ptDof, &lapack_info);

		algebra::vector<double> K2(tmp0 + tmp1);

		_x = x + xd * (0.5 * dt) + K2 * (0.125 * dt * dt);
		_xd = xd + K2 * (0.5 * dt); _M = M; _K = K; _C = C;
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _K.getDataPointer(), &ptDof, _x.get_ptr(), &lapack_one, &lapack_zero, tmp0.get_ptr(), &lapack_one);
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _C.getDataPointer(), &ptDof, _xd.get_ptr(), &lapack_one, &lapack_zero, tmp1.get_ptr(), &lapack_one);
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp0.get_ptr(), &ptDof, &lapack_info); _M = M;
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp1.get_ptr(), &ptDof, &lapack_info);

		algebra::vector<double> K3(tmp0 + tmp1);

		_x = x + xd * dt + K3 * (0.5 * dt * dt);
		_xd = xd + K3 * dt; _M = M; _K = K; _C = C;
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _K.getDataPointer(), &ptDof, _x.get_ptr(), &lapack_one, &lapack_zero, tmp0.get_ptr(), &lapack_one);
		dgemv_(&trans, &ptDof, &ptDof, &lapack_mone, _C.getDataPointer(), &ptDof, _xd.get_ptr(), &lapack_one, &lapack_zero, tmp1.get_ptr(), &lapack_one);
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp0.get_ptr(), &ptDof, &lapack_info); _M = M;
		dgesv_(&ptDof, &lapack_one, _M.getDataPointer(), &ptDof, permutation, tmp1.get_ptr(), &ptDof, &lapack_info);

		algebra::vector<double> K4(tmp0 + tmp1);
		
		x_i_end = x(9);
		xd_i_end = xd(9);
		//vector<double> tt = (K1 + K2 + K3) * (1 / 6);
		x = x + xd * dt + (K1 + K2 + K3) * (1.0 / 6.0) * dt * dt;
		xd = xd + (K1 + (K2 * 2) + (K3 * 2) + K4) * (1.0 / 6.0) * dt;

		if (x(9) >= -pComp * 0.001){
			v_ep = ((-pComp * 0.001 - x_i_end) * (xd(9) - xd_i_end) / (x(9) - x_i_end)) + xd_i_end;
			break;
		}
	}

	delete[] permutation; permutation = NULL;
	return v_ep;
}

bool dh_solver::initialize()
{
	if (FreeLength < InstallLength)
	{
		QMessageBox msgbox;
		msgbox.setText(kor("장착길이 입력값이 자유장 길이를 초과합니다."));
		msgbox.setIcon(QMessageBox::Critical);
		msgbox.exec();
		return false;
	}
	
	cMinStiffness = 0.8;
	cMaxStiffness = 3;
	cMinStress = cMaxStress * 0.8;
	cWeight = 120;
	cMass = 50;
	PreComp = FreeLength - InstallLength;
	TotalComp = PreComp + Stroke;
	BHLimit = FreeLength - (TotalComp / DisBH);

	N_Lower = N + NLower;
	N_Upper = N + NUpper;
	D_Lower = D + DLower;
	D_Upper = D + DUpper;
	d_Lower = d + dLower;
	d_Upper = d + dUpper;
	Free_length_lower = FreeLength + FreeLengthLower;
	Free_length_upper = FreeLength + FreeLengthUpper;

	// Reference 스프링의 강성, 질량, 응력 등 계산
	k_ref = (shearModulus * pow(d, 4)) / (8 * N * pow(D, 3));
	Mass = density * (pow(M_PI, 2) * pow(d, 2) * D * (N + 1.6)) / 4;
	Mass_EB = density * (pow(M_PI, 2) * pow(d, 2) * D * 0.8) / 4;
	C = D / d;
	AR = FreeLength / D;
	B_Height = (N + 1.6) * d;
	P = k_ref * TotalComp;
	P_BH = k_ref * (FreeLength - B_Height);
	Kw = (4 * C - 1) / (4 * C - 4) + 0.615 / C;
	Sc = (8 * D * P * Kw) / (M_PI * pow(d, 3));
	Sc_BH = (8 * D * P_BH * Kw) / (M_PI * pow(d, 3));
	PE_full_ref = 0.5 * k_ref * 9.81 * 1000 * pow(TotalComp * 0.001, 2);
	PE_init_ref = 0.5 * k_ref * 9.81 * 1000 * pow(PreComp * 0.001, 2);
	PE_act_ref = PE_full_ref - PE_init_ref;

	//if ((FreeLength - TotalComp) <= B_Height){
	//	QMessageBox msgbox;
	//	msgbox.setText(kor("밀착고가));
	//	msgbox.setIcon(QMessageBox::Critical);
	//	msgbox.exec();
	//}

	v_ep = lumped(k_ref, Mass, Mass_EB, eq_mass, TotalComp, PreComp);

	transfer_energy_ref = 0.5 * eq_mass * pow(v_ep, 2);
	efficiency_ref = transfer_energy_ref / PE_act_ref;


	ref_result.N = N;
	ref_result.D = D;
	ref_result.d = d;
	ref_result.FreeLength = FreeLength;
	ref_result.k = k_ref;
	ref_result.Mass = Mass;
	ref_result.B_Height = B_Height;
	ref_result.P = P;
	ref_result.P_BH = P_BH;
	ref_result.Sc = Sc;
	ref_result.Sc_BH = Sc_BH;
	ref_result.PE_act = PE_act_ref;
	ref_result.Efficiency = efficiency_ref;
	ref_result.transferEnergy = transfer_energy_ref;
	ref_result.v_ep = v_ep;
	ref_result.C = C;
	ref_result.AR = AR;

	double dL = Free_length_upper - Free_length_lower;
	double dN = N_Upper - N_Lower;
	double dD = D_Upper - D_Lower;
	double dd = d_Upper - d_Lower;
	double prod = dL*dN*dD*dd;
	if (prod <= 0){
		QString str;
		if (dL <= 0) str = kor("자유장 설계변수");
		else if (dN <= 0) str = kor("유효권수 설계변수");
		else if (dD <= 0) str = kor("중심경 설계변수");
		else if (dd <= 0) str = kor("재질경 설계변수");
		QMessageBox msgbox;
		msgbox.setText(str + kor(" 입력에 잘못된 값이 있습니다."));
		msgbox.setIcon(QMessageBox::Critical);
		msgbox.exec();
		return false;
	}
	nL_iter = (unsigned int)floor((Free_length_upper - Free_length_lower) / deltaFreeLength) + 1;
	nN_iter = (unsigned int)floor((N_Upper - N_Lower) / deltaN) + 1;
	nD_iter = (unsigned int)floor((D_Upper - D_Lower) / deltaD) + 1;
	nd_iter = (unsigned int)floor((d_Upper - d_Lower) / deltad) + 1;
	total_iter = nL_iter * nN_iter * nD_iter * nd_iter;

	QFile file("autoSave.dat");
	if (file.open(QIODevice::WriteOnly)){
		file.write((char*)&N, sizeof(double) * 39);
	}
	file.close();
	return true;
}

bool dh_solver::solve()
{
	double cMinPotential = 1.0 - cPotential * 0.01;
	double cMaxPotential = 1.0 + cPotential * 0.01;
	double cweight = cWeight * 0.01;
	unsigned count = 0;
	success_count = 0;
	//dhs->setProgressBarMaximum(total_iter);
	double k, PE_full, PE_init, PE_act;
	for (unsigned int iL = 0; iL < nL_iter; iL++){
		FreeLength = Free_length_lower + iL * deltaFreeLength;
		for (unsigned int iN = 0; iN < nN_iter; iN++){
			N = N_Lower + iN * deltaN;
			for (unsigned int iD = 0; iD < nD_iter; iD++){
				D = D_Lower + iD * deltaD;
				for (unsigned int id = 0; id < nd_iter; id++){
					d = d_Lower + id * deltad;
					PreComp = FreeLength - InstallLength;
					TotalComp = PreComp + Stroke;
					BHLimit = FreeLength - (TotalComp / DisBH);
					k = (shearModulus * pow(d, 4)) / (8 * N * pow(D, 3));
					Mass = density * (pow(M_PI, 2) * pow(d, 2) * D * (N + 1.6)) / 4.0;
					Mass_EB = density * (pow(M_PI, 2) * pow(d, 2) * D * 0.8) / 4.0;
					C = D / d;
					AR = FreeLength / D;
					B_Height = (N + 1.6) * d;
					Kw = (4 * C - 1) / (4 * C - 4) + 0.615 / C;
					P = k * TotalComp;
					P_BH = k * (FreeLength - B_Height);
					Sc = (8 * D * P * Kw) / (M_PI * pow(d, 3));
					Sc_BH = (8 * D * P_BH * Kw) / (M_PI * pow(d, 3));
					PE_full = 0.5 * k * 9.81 * 1000 * pow(TotalComp * 1e-3, 2);
					PE_init = 0.5 * k * 9.81 * 1000 * pow(PreComp * 1e-3, 2);
					PE_act = PE_full - PE_init;
					count = count + 1;
					
					if (Sc <= cMaxStress && Sc >= cMinStress && Sc_BH <= cBHStress && B_Height <= BHLimit && k >= k_ref * cMinStiffness && k <= k_ref * cMaxStiffness && Mass <= cMass && PE_act >= PE_act_ref * cMinPotential && PE_act <= PE_act_ref * cMaxPotential && P <= cweight * ref_result.P && C >= cMinSpringIndex && C <= cMaxSpringIndex && AR <= cAspectRatio)
					{
						v_ep = lumped(k, Mass, Mass_EB, eq_mass, TotalComp, PreComp);
						double transfer_energy = 0.5 * eq_mass * pow(v_ep, 2);
						double efficiency = transfer_energy / PE_act;

						resultSet result = { N, D, d, FreeLength, k, Mass, B_Height, P, P_BH, Sc, Sc_BH, PE_act, efficiency, transfer_energy, v_ep, C, AR };
						results.push(result);
						success_count++;
						emit mySignal((int)count);
					}
				}
			}
		}
	}
	emit mySignal((int)count);
	results.adjustment();
	return true;
}