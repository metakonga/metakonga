#include "cu_setOfParticle.h"

cu_setOfParticle::cu_setOfParticle()
	: h_pos(0)
	, d_pos(0)
	, h_vel(0)
	, d_vel(0)
	, h_acc(0)
	, d_acc(0)
	, h_eup(0)
	, d_eup(0)
	, d_euv(0)
	, d_eua(0)
	, d_omega(0)
	, d_global_omega(0)
	, d_alpha(0)
	, d_force(0)
	, d_moment(0)
	, partIdx(0)
	, nResultFrame(0)
{

}

cu_setOfParticle::cu_setOfParticle(unsigned np)
	: h_pos(0)
	, d_pos(0)
	, h_vel(0)
	, d_vel(0)
	, h_acc(0)
	, d_acc(0)
	, h_eup(0)
	, d_eup(0)
	, d_euv(0)
	, d_eua(0)
	, d_omega(0)
	, d_alpha(0)
	, d_force(0)
	, d_moment(0)
	, partIdx(0)
	, nResultFrame(0)
{
	h_pos = new float[np*3];
}

cu_setOfParticle::~cu_setOfParticle()
{
	if(h_pos) delete [] h_pos; h_pos=0;
	if(h_vel) delete [] h_vel; h_vel=0;
	if(h_eup) delete [] h_eup; h_eup=0;
	if(h_acc) delete [] h_acc; h_acc=0;
	if(d_pos) cudaFree(d_pos); d_pos=0;
	if(d_vel) cudaFree(d_vel); d_vel=0;
	if(d_acc) cudaFree(d_acc); d_acc=0;
	if(d_eup) cudaFree(d_eup); d_eup=0;
	if(d_euv) cudaFree(d_euv); d_euv=0;
	if(d_eua) cudaFree(d_eua); d_eua=0;
	if(d_omega) cudaFree(d_omega); d_omega=0;
	if(d_alpha) cudaFree(d_alpha); d_alpha=0;
	if(d_global_omega) cudaFree(d_global_omega); d_global_omega=0;
	if(d_force) cudaFree(d_force); d_force=0;
	if(d_moment) cudaFree(d_moment); d_moment=0;
}

void cu_setOfParticle::allocDeviceMemory()
{
	checkCudaErrors( cudaMalloc((void**)&d_pos, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_vel, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_acc, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_eup, sizeof(float)*nP*4) );
	checkCudaErrors( cudaMalloc((void**)&d_euv, sizeof(float)*nP*4) );
	checkCudaErrors( cudaMalloc((void**)&d_eua, sizeof(float)*nP*4) );
	checkCudaErrors( cudaMalloc((void**)&d_omega, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_alpha, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_global_omega, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_force, sizeof(float)*nP*3) );
	checkCudaErrors( cudaMalloc((void**)&d_moment, sizeof(float)*nP*3) );


	checkCudaErrors( cudaMemcpy(d_pos, h_pos, sizeof(float)*nP*3, cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(d_vel, h_vel, sizeof(float)*nP*3, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_eup, h_eup, sizeof(float)*nP*4, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_acc, h_acc, sizeof(float)*nP*3, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMemset(d_vel,0,sizeof(float)*nP*3) );
	checkCudaErrors( cudaMemset(d_euv,0,sizeof(float)*nP*4) );
	checkCudaErrors( cudaMemset(d_eua,0,sizeof(float)*nP*4) );
	checkCudaErrors( cudaMemset(d_omega,0,sizeof(float)*nP*3) );
	checkCudaErrors( cudaMemset(d_alpha,0,sizeof(float)*nP*3) );
	checkCudaErrors( cudaMemset(d_global_omega,0,sizeof(float)*nP*3) );
	checkCudaErrors( cudaMemset(d_force,0,sizeof(float)*nP*3) );
	checkCudaErrors( cudaMemset(d_moment,0,sizeof(float)*nP*3) );
}

void cu_setOfParticle::arrange_particle(float3& sp, float3& ep, float radius, float scale)
{
	uint3 n_p;
	unsigned p_id=0;
	n_p.x=static_cast<unsigned>(abs((ep.x-sp.x)/(radius*scale)));
	n_p.y=static_cast<unsigned>(abs((ep.y-sp.y)/(radius*scale)));
	n_p.z=static_cast<unsigned>(abs((ep.z-sp.z)/(radius*scale)));
	if(n_p.x==0) n_p.x=1;
	if(n_p.y==0) n_p.y=1;
	if(n_p.z==0) n_p.z=1;
	nP=n_p.x*n_p.y*n_p.z;
	h_pos = new float[nP*3];
	memset(h_pos,0,sizeof(float)*nP*3);
	//h_vel = new float[nP*3];
	//memset(h_vel,0,sizeof(float)*nP*3);
	h_eup = new float[nP*4];
	memset(h_eup,0,sizeof(float)*nP*4);
	h_acc = new float[nP*3];
	memset(h_acc,0,sizeof(float)*nP*3);
// 	h_pos[0]=0.0f;
// 	h_pos[1]=-0.1885f;
// 	//h_vel[0]=0.4f;
// 	h_eup[0]=1.0f;
// 	h_pos[3]=0.0f;
// 	h_pos[4]=-0.195f;
// 	h_eup[4]=1.0f;

	srand(1973);
	float jitter=radius*0.001f;
	for(unsigned z=0; z<n_p.z; z++){
		for(unsigned y=0; y<n_p.y; y++){
			for(unsigned x=0; x<n_p.x; x++){
				h_pos[p_id*3+0]=(sp.x+radius+x*scale*radius)+frand()*jitter;//(base_type)(frand()*(2.0-1.0)*jitter);
				h_pos[p_id*3+1]=(sp.y+radius+y*scale*radius)+frand()*jitter;//(base_type)(frand()*(2.0-1.0)*jitter);
				h_pos[p_id*3+2]=(sp.z+radius+z*scale*radius)+frand()*jitter;//(base_type)(frand()*(2.0-1.0)*jitter);
				h_eup[p_id*4+0]=1.0f; h_eup[p_id*4+1]=0.0f; h_eup[p_id*4+2]=0.0f; h_eup[p_id*4+3]=0.0f;
				p_id++;
			}
		}
	}
}

void cu_setOfParticle::exportResult()
{
	checkCudaErrors( cudaMemcpy(h_pos, d_pos, sizeof(float)*nP*3, cudaMemcpyDeviceToHost) );
	sprintf_s(partName,"PART%d.dat", partIdx);
	sprintf_s(resultDirectory,"C:/opengmap/result/particle/");
	strcat_s(resultDirectory,sizeof(char)*256,partName);
	fopen_s(&out,resultDirectory,"wb");
	fwrite(h_pos, sizeof(float),nP*3,out);
	nResultFrame++;
	partIdx++;
	fclose(out);
}
