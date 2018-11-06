#include "colors.h"
#include "vcontroller.h"

using namespace ucolors;

double* colorMap::tmp_color = NULL;

colorMap::colorMap()
	: cmt(COLORMAP_PRESSURE)
	, limits(NULL)
	, min_vx(NULL)
	, min_vy(NULL)
	, min_vz(NULL)
	, max_vx(NULL)
	, max_vy(NULL)
	, max_vz(NULL)
	, min_p(NULL)
	, max_p(NULL)
{
	initColormap(1);
	// 	limits[0] = -0.23;
	// 	limits[1] = -0.15;
	// 	limits[2] = -0.07;
	// 	limits[3] = 0.02;
	// 	limits[4] = 0.10;
	// 	limits[5] = 0.18;
	// 	limits[6] = 0.26;
	// 	limits[7] = 0.34;
	// 	limits[8] = 0.43;
	// 	limits[9] = 0.51;
	// 	limits[10] = 0.59;
	// 	limits[11] = 0.67;
	// 	limits[12] = 0.76;
	// 	limits[13] = 0.84;
	// 	limits[14] = 0.92;
	// 	limits[15] = 0.92;
}

colorMap::colorMap(size_t _npart)
	: cmt(COLORMAP_PRESSURE)
	, limits(NULL)
	, min_vx(NULL)
	, min_vy(NULL)
	, min_vz(NULL)
	, max_vx(NULL)
	, max_vy(NULL)
	, max_vz(NULL)
	, min_p(NULL)
	, max_p(NULL)
	, npart(_npart)
{
	initColormap(npart);
}

colorMap::~colorMap()
{
	clearMemory();
}

void colorMap::clearMemory()
{
	if (limits) delete[] limits; limits = NULL;
	if (min_vx) delete[] min_vx; min_vx = NULL;
	if (min_vy) delete[] min_vy; min_vy = NULL;
	if (min_vz) delete[] min_vz; min_vz = NULL;
	if (max_vx) delete[] max_vx; max_vx = NULL;
	if (max_vy) delete[] max_vy; max_vy = NULL;
	if (max_vz) delete[] max_vz; max_vz = NULL;
	if (min_p) delete[] min_p; min_p = NULL;
	if (max_p) delete[] max_p; max_p = NULL;
	if (tmp_color) delete[] tmp_color; tmp_color = NULL;
}

void colorMap::initColormap(size_t npart)
{
	c[0][0] = 0; c[0][1] = 0, c[0][2] = 255;
	c[1][0] = 0; c[1][1] = 63, c[1][2] = 255;
	c[2][0] = 0; c[2][1] = 127, c[2][2] = 255;
	c[3][0] = 0; c[3][1] = 191, c[3][2] = 255;
	c[4][0] = 0; c[4][1] = 255, c[4][2] = 255;
	c[5][0] = 0; c[5][1] = 255, c[5][2] = 191;
	c[6][0] = 0; c[6][1] = 255, c[6][2] = 127;
	c[7][0] = 0; c[7][1] = 255, c[7][2] = 63;
	c[8][0] = 0; c[8][1] = 255, c[8][2] = 0;
	c[9][0] = 63; c[9][1] = 255, c[9][2] = 0;
	c[10][0] = 127; c[10][1] = 255, c[10][2] = 0;
	c[11][0] = 191; c[11][1] = 255, c[11][2] = 0;
	c[12][0] = 255; c[12][1] = 255, c[12][2] = 0;
	c[13][0] = 255; c[13][1] = 191, c[13][2] = 0;
	c[14][0] = 255; c[14][1] = 127, c[14][2] = 0;
	c[15][0] = 255; c[15][1] = 63, c[15][2] = 0;
	c[16][0] = 255; c[16][1] = 0, c[16][2] = 0;
	clearMemory();
	min_vx = new double[npart];
	min_vy = new double[npart];
	min_vz = new double[npart];
	max_vx = new double[npart];
	max_vy = new double[npart];
	max_vz = new double[npart];
	min_p = new double[npart];
	max_p = new double[npart];
	limits = new double[npart * 16 * COLORMAP_ENUMS];
	memset(min_vx, 0, sizeof(double) * npart);
	memset(min_vy, 0, sizeof(double) * npart);
	memset(min_vz, 0, sizeof(double) * npart);
	memset(max_vx, 0, sizeof(double) * npart);
	memset(max_vy, 0, sizeof(double) * npart);
	memset(max_vz, 0, sizeof(double) * npart);
	memset(min_p, 0, sizeof(double) * npart);
	memset(max_p, 0, sizeof(double) * npart);
	memset(limits, 0, sizeof(double) * 16 * COLORMAP_ENUMS);
}

double colorMap::getLimit(int i)
{
	size_t cframe = vcontroller::getFrame();
	size_t idx = (npart * 16) * cmt + cframe * 16;
	return limits[idx + i];
}

void colorMap::setColormap(QColor *clist, double* lim)
{
	for (int i = 0; i < 17; i++)
	{
		c[i][0] = clist[i].red();
		c[i][1] = clist[i].green();
		c[i][2] = clist[i].blue();
	}
	size_t cframe = vcontroller::getFrame();
	size_t idx = cframe * 16;
	for (int i = 0; i < 16; i++)
	{
		limits[idx + i] = lim[i];
	}
}

void colorMap::setLimitsFromMinMax()
{
	size_t cframe = vcontroller::getFrame();
	size_t idx = 0;// cframe * 16;
	for (size_t i = 0; i < npart; i++)
	{
		idx = i * 16;
		if (cmt == COLORMAP_PRESSURE)
		{
			double range = max_p[cframe] - min_p[cframe];
			double dp = range / 18;
			for (int i = 1; i < 17; i++){
				limits[idx + (i - 1)] = min_p[cframe] + dp * i;
			}
		}
		else if (cmt == COLORMAP_VELOCITY_X)
		{
			double range = max_vx[cframe] - min_vx[cframe];
			double dv = range / 18;
			for (int i = 1; i < 17; i++){
				limits[idx + (i - 1)] = min_vx[cframe] + dv * i;
			}
		}
	}
}

void colorMap::getColorRamp(size_t ci, double v, double *clr)
{
	//size_t cframe = vcontroller::getFrame();
	size_t idx = (npart * 16) * cmt + ci * 16;
	int t = 0;
	double div = 1.0 / 255;
	if (v <= limits[idx + 0])
	{
		clr[0] = c[0][0] * div; clr[1] = c[0][1] * div; clr[2] = c[0][2] * div;
		return;
	}
	else if (v >= limits[idx + 15])
	{
		clr[0] = c[16][0] * div; clr[1] = c[16][1] * div; clr[2] = c[0][2] * div;
		return;
	}

	for (int i = 1; i < 16; i++){
		if (v <= limits[idx + i]){
			t = i;
			break;
		}
	}
	clr[0] = c[t][0] * div; clr[1] = c[t][1] * div; clr[2] = c[t][2] * div;
}

void colorMap::setMinMax(size_t cpart, double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8)
{
	min_vx[cpart] = v1;
	min_vy[cpart] = v2;
	min_vz[cpart] = v3;
	max_vx[cpart] = v4;
	max_vy[cpart] = v5;
	max_vz[cpart] = v6;
	min_p[cpart] = v7;
	max_p[cpart] = v8;

	double range = max_p[cpart] - min_p[cpart];
	double dp = range / 18;
	size_t idx = cpart * 16;
	for (int i = 1; i < 17; i++){
		limits[idx + (i - 1)] = min_p[cpart] + dp * i;
	}
	range = max_vx[cpart] - min_vx[cpart];
	double dv = range / 18;
	idx = (npart * 16) + cpart * 16;
	for (int i = 1; i < 17; i++){
		limits[idx + (i - 1)] = min_vx[cpart] + dv * i;
	}
}

double* ucolors::colorMap::particleColorBySphType(unsigned int np, unsigned int* tp)
{
	if(!tmp_color)
		tmp_color = new double[np * 4];
	for (unsigned int i = 0; i < np; i++){
		// 		if (tp[i] != FLUID)
		// 			continue;
		unsigned int pid = i * 3;
		unsigned int cid = i * 4;
		if (tp[i] == 1)
		{
			tmp_color[cid + 0] = 0.0f;
			tmp_color[cid + 1] = 0.0f;
			tmp_color[cid + 2] = 1.0f;
			tmp_color[cid + 3] = 1.0f;
			//nfluid++;
		}
		else if (tp[i] == 3)
		{
			tmp_color[cid + 0] = 1.0f;
			tmp_color[cid + 1] = 0.0f;
			tmp_color[cid + 2] = 0.0f;
			tmp_color[cid + 3] = 1.0f;
		}
		else if (tp[i] == 4)
		{
			tmp_color[cid + 0] = 0.0f;
			tmp_color[cid + 1] = 1.0f;
			tmp_color[cid + 2] = 0.0f;
			tmp_color[cid + 3] = 1.0f;
		}
		else if (tp[i] == 2)
		{
			tmp_color[cid + 0] = 0.0f;
			tmp_color[cid + 1] = 1.0f;
			tmp_color[cid + 2] = 1.0f;
			tmp_color[cid + 3] = 1.0f;
		}
	}
	return tmp_color;
}
