#ifndef COLORS_H
#define COLORS_H

#include <QList>
#include <QColor>

namespace ucolors
{
	enum colorMapTarget{ COLORMAP_PRESSURE = 0, COLORMAP_VELOCITY_X, COLORMAP_VELOCITY_Y, COLORMAP_VELOCITY_Z, COLORMAP_VELOCITY_MAG, COLORMAP_ENUMS/*, COLORMAP_VELOCITY_Y, COLORMAP_VELOCITY_Z*/ };
	class colorMap
	{
	public:
		colorMap();
		colorMap(size_t _npart);
		~colorMap();

		void clearMemory();
		void setColormap(QColor *clist, double *lim);
		void initColormap(size_t npart);
		QColor getColor(int i) { return QColor(c[i][0], c[i][1], c[i][2]); }
		double getLimit(int i);// { return limits[i]; }
		void setTarget(colorMapTarget _cmt) { cmt = _cmt; }
		void setMinMax(size_t cpart, double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8);
		void getColorRamp(size_t c, double v, double *clr);
		void setLimitsFromMinMax();
		void setNumParts(size_t _npart) { npart = _npart; }
		colorMapTarget target() { return cmt; }
		
		static double* particleColorBySphType(unsigned int np, unsigned int* tp);

	private:
		static double* tmp_color;
		size_t npart;
		double *limits;
		double c[17][3];
		colorMapTarget cmt;

		double *min_vx;
		double *min_vy;
		double *min_vz;
		double *max_vx;
		double *max_vy;
		double *max_vz;
		double *min_p;
		double *max_p;

	};
	// 	void colorRamp(double t, double *r)
	// 	{
	// 		double div255 = 1.0f / 255.0f;
	// 		const int ncolors = 10;
	// 
	// 		double c[ncolors][3] = {
	// 			{ 3 * div255, 0.0f, 102 * div255, },
	// 			{ 0 * div255, 84 * div255, 1.0f, },
	// 			{ 0.0f, 216.0f * div255, 1.0f, },
	// 			{ 29.0f * div255, 219.0f * div255, 22.0f * div255, },
	// 			{ 190.0f * div255, 1.0f, 0.0f },
	// 			{ 171.0f * div255, 242 * div255, 0.0f, },
	// 			{ 1.0f, 228.0f * div255, 0.0f, },
	// 			{ 1.0f, 187 * div255, 0.0f, },
	// 			{ 1.0f, 94.0f * div255, 0.0f, },
	// 			{ 1.0f, 0.0f, 0.0f }
	// 		};
	// 		int i = (int)t;
	// 		if (i < 0 || i > 9){
	// 			r[0] = 1.0f;
	// 			r[1] = 0.0f;
	// 			r[2] = 0.0f; return;
	// 		}
	// 		r[0] = c[i][0];
	// 		r[1] = c[i][1];
	// 		r[2] = c[i][2];
	// 	}
	// 	void lid_color_ramp(double t, double *r)
	// 	{
	// 		double div255 = 1.0f / 255.0f;
	// 		const int ncolors = 10;
	// 		if (t >= -0.30 && t < -0.23){ r[0] = 0; r[1] = 0; r[2] = 1; }
	// 		else if (t >= -0.23 && t < -0.15) { r[0] = 0.0; r[1] = 0.2; r[2] = 1.0; }
	// 		else if (t >= -0.15 && t < -0.07) { r[0] = 0.0; r[1] = 0.4; r[2] = 1.0; }
	// 		else if (t >= -0.07 && t < 0.02)  { r[0] = 0.0; r[1] = 0.8; r[2] = 1.0; }
	// 		else if (t >= 0.02 && t < 0.10)   { r[0] = 0.0; r[1] = 1.0; r[2] = 0.8; }
	// 		else if (t >= 0.10 && t < 0.18)   { r[0] = 0.0; r[1] = 1.0; r[2] = 0.6; }
	// 		else if (t >= 0.18 && t < 0.26)   { r[0] = 0.0; r[1] = 1.0; r[2] = 0.4; }
	// 		else if (t >= 0.26 && t < 0.34)   { r[0] = 0.0; r[1] = 1.0; r[2] = 0.2; }
	// 		else if (t >= 0.34 && t < 0.43)   { r[0] = 0.2; r[1] = 1.0; r[2] = 0.0; }
	// 		else if (t >= 0.43 && t < 0.51)   { r[0] = 0.4; r[1] = 1.0; r[2] = 0.0; }
	// 		else if (t >= 0.51 && t < 0.59)   { r[0] = 0.6; r[1] = 1.0; r[2] = 0.0; }
	// 		else if (t >= 0.59 && t < 0.67)   { r[0] = 1.0; r[1] = 1.0; r[2] = 0.0; }
	// 		else if (t >= 0.67 && t < 0.76)   { r[0] = 1.0; r[1] = 0.8; r[2] = 0.0; }
	// 		else if (t >= 0.76 && t < 0.84)   { r[0] = 1.0; r[1] = 0.4; r[2] = 0.0; }
	// 		else if (t >= 0.84 && t < 0.92)   { r[0] = 1.0; r[1] = 0.2; r[2] = 0.0; }
	// 		else if (t >= 0.92)				  { r[0] = 1.0; r[1] = 0.0; r[2] = 0.0; }
	// 	}

}

#endif