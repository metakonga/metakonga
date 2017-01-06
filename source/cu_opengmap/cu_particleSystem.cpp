#include "cu_particleSystem.h"

cu_particleSystem::cu_particleSystem()
{

}

cu_particleSystem::~cu_particleSystem()
{
	std::fstream fo("C:/opengmap/result/particle/particleViewInfo.txt", std::ios::out);
	fo << "particle" << std::endl;
	fo << "nObject" << " " << h_paras.nParticle << std::endl;
	fo << "Radius" << " " << h_paras.radius << std::endl;
	fo << "nFrame" << " " << ps->nResultFrame << std::endl;
	fo << "outputStep" << " " << output_step << std::endl;
	fo.close();
	if(ps) delete ps; ps=0;
}

void cu_particleSystem::initialize()
{
	load_input("C:/C++/kangsia/common/inc/cu_opengmap/indata.txt");
	load_boundary("C:/C++/kangsia/common/inc/cu_opengmap/modelingData.txt");
	setSymbolicParameter(&h_paras);
	checkCudaErrors( cudaMalloc((void**)&d_boundaries, sizeof(boundaryType)*h_paras.nBoundary) );
	checkCudaErrors( cudaMemcpy(d_boundaries, boundaries, sizeof(boundaryType)*h_paras.nBoundary, cudaMemcpyHostToDevice) );
	ps->allocDeviceMemory();

	std::cout << "The Number Of Particle : " << h_paras.nParticle << std::endl;
}

void cu_particleSystem::run()
{
	/*double sumationTime=0;*/
	float MAX_Y=-0.25f;
	float A_MAX_Y[10] = {-0.25f,-0.25f,-0.25f,-0.25f,-0.25f,-0.25f,-0.25f,-0.25f,-0.25f,-0.25f};
	float T_PARA[10] = {-0.5f,-0.4f,-0.3f,-0.2f,-0.1f,0.0f,0.1f,0.2f,0.3f,0.4f};
	tm.start();
	while(cframe++ != nframe)
	{
		std::cout << "cframe : " << cframe << std::endl;
		exeEvent(cframe);
		setEulerParameters(
			ps->d_eup, 
			ps->d_euv, 
			ps->d_eua, 
			ps->d_omega, 
			/*ps->d_global_omega, */
			ps->d_alpha, 
			h_paras.nParticle);
		if(!((cframe-1)%1000))
		{
			std::cout << "duration time : " << cframe*h_paras.dt << std::endl;
		}
		updatePosition(ps->d_pos, ps->d_vel, ps->d_acc, ps->d_eup, ps->d_euv, ps->d_eua, h_paras.nParticle);
// 		if(!((cframe-1)%output_frame))
// 			ps->exportResult();

		if(cframe == 45000)
		{
			//float MAX_Y=-0.25f;
			checkCudaErrors( cudaMemcpy(ps->h_pos, ps->d_pos, sizeof(float)*h_paras.nParticle*3, cudaMemcpyDeviceToHost) );
			for(unsigned i(0); i < h_paras.nParticle; i++)
			{
				if(ps->h_pos[i*3+1] > MAX_Y)
				{
					MAX_Y = ps->h_pos[i*3+1];
				}
			}
		}



		calculateHashAndIndex(d_GridParticleHash, d_GridParticleIndex, ps->d_pos, h_paras.nParticle);

		reorderDataAndFindCellStart(
			d_sortedPos,
			d_sortedVel,
			d_sortedOmega,
			ps->d_pos,
			ps->d_vel,
			ps->d_omega,
			d_CellStart, 
			d_CellEnd, 
			d_GridParticleHash, 
			d_GridParticleIndex, 
			h_paras.nParticle, 
			h_paras.nCell);

		calculateCollideForce(
			ps->d_vel,
			ps->d_global_omega,
			d_sortedPos,
			d_sortedVel,
			d_sortedOmega,
			d_boundaries,
			ps->d_force,
			ps->d_moment,
			d_GridParticleIndex,
			d_CellStart,
			d_CellEnd,
			h_paras.nParticle,
			h_paras.nCell);


		updateVelocity(
			ps->d_vel,
			ps->d_acc,
			ps->d_omega,
			ps->d_alpha,
			ps->d_force,
			ps->d_moment, 
			h_paras.nParticle);
	}
	//std::cout << "sumationTime : " << sumationTime << std::endl;
	tm.end();
	tm.durationTime();
	std::cout << MAX_Y << std::endl;
	checkCudaErrors( cudaMemcpy(ps->h_pos, ps->d_pos, sizeof(float)*h_paras.nParticle*3, cudaMemcpyDeviceToHost) );

	for(unsigned i(0); i < h_paras.nParticle; i++)
	{
		for(unsigned j(0); j < 10 ;j++)
		{
			if((ps->h_pos[i*3] > T_PARA[j]) & (ps->h_pos[i*3]< T_PARA[j]+0.005f))
			{
				if(ps->h_pos[i*3 + 1] > A_MAX_Y[j])
				{
					A_MAX_Y[j] = ps->h_pos[i*3 +1];
				}
			}
		}

	}
	for(unsigned k(0); k < 10; k++)
	{
		std::cout << A_MAX_Y[k]<<std::endl;
	}
	
}