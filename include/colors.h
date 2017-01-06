#ifndef COLORS_H
#define COLORS_H

//#include "algebra/vector4.hpp"

namespace ucolors
{
	void colorRamp(float t, float *r)
	{
		float div255 = 1.0f / 255.0f;
		const int ncolors = 9;

		float c[ncolors][3] = {
			{ 3 * div255, 0.0f, 102 * div255, },
			{ 0 * div255, 84 * div255, 1.0f, },
			{ 0.0f, 216.0f * div255, 1.0f, },
			{ 29.0f * div255, 219.0f * div255, 22.0f * div255, },
			{ 171.0f * div255, 242 * div255, 0.0f, },
			{ 1.0f, 228.0f * div255, 0.0f, },
			{ 1.0f, 187 * div255, 0.0f, },
			{ 1.0f, 94.0f * div255, 0.0f, },
			{ 1.0f, 0.0f, 0.0f }
		};
		int i = (int)t;
		if (i < 0 || i > 9){
			r[0] = 1.0f;
			r[1] = 0.0f;
			r[2] = 0.0f; return;
		}
		r[0] = c[i][0];
		r[1] = c[i][1];
		r[2] = c[i][2];
	}

// 	enum color_type{BLUE=0, GREEN, RED, BLACK, WHITE};
// 	static algebra::vector4<float> GetColor(color_type c)
// 	{
// 		algebra::vector4<float> clr;
// 		switch (c)
// 		{
// 		case BLUE: clr = algebra::vector4<float>(0.0f, 0.0f, 1.0f, 1.0f); break;
// 		case GREEN: clr = algebra::vector4<float>(0.0f, 1.0f, 0.0f, 1.0f); break;
// 		case RED: clr = algebra::vector4<float>(1.0f, 0.0f, 0.0f, 1.0f); break;
// 		case BLACK: clr = algebra::vector4<float>(0.0f, 0.0f, 0.0f, 1.0f); break;
// 		case WHITE: clr = algebra::vector4<float>(1.0f, 1.0f, 1.0f, 1.0f); break;
// 		}
// 		return clr;
// 	}
}

#endif