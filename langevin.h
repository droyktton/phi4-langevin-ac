// Langevin (stochastic Euler) solver for the disordered phi^4 model
// driven by an AC field.
//
// Model:
//   dphi/dt = c*Laplacian(phi) + eps0*[r0*(1+r(x,y))*phi - phi^3] + h(t) + eta
// with r(x,y) uncorrelated random-bond disorder, uniform in [-Delta,Delta],
// h(t) = h0*cos(2*pi*f*t), and Gaussian white noise
// <eta(x,t) eta(x',t')> = 2T delta(x-x') delta(t-t'), on an LxL grid with
// periodic boundary conditions in both x and y.
//
// Integration: explicit Euler-Maruyama,
//   phi_new = phi + dt*F(phi,h(t)) + sqrt(2*T*dt)*xi,  xi ~ N(0,1),
// as a Jacobi update (two buffers) so every site is updated in parallel.
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// model coefficients (c=eps0=r0=1 without loss of generality)
#ifndef CEL
#define CEL	1.0	// elastic constant c
#endif
#ifndef EPSILON0
#define EPSILON0	1.0	// eps0
#endif
#ifndef R0
#define R0	1.0	// bare (undisordered) coefficient of phi
#endif

/* counter-based random numbers, used both for the quenched disorder
   r(x,y) and for the thermal noise (a reproducible, independent stream
   per site and per time step): http://www.thesalmons.org/john/random123/ */
#include <Random123/philox.h>
#include <Random123/u01.h>
typedef r123::Philox2x32 RNG;
typedef r123::Philox4x32 NoiseRNG;

#define NOISE_KEY_TAG	0x4E4F4953u	// "NOIS": separates the noise stream from the disorder one

// quenched random-bond disorder r(i,j), uniform in [-delta,delta],
// generated from a counter-based RNG keyed by the site index and the
// disorder seed, so a given (seed,site) always gives the same value.
__device__
float site_disorder(int site_index, unsigned long disorder_seed, float delta)
{
	RNG rng;
	RNG::ctr_type c={{}};
	RNG::key_type k={{}};
	c[1]=uint32_t(disorder_seed);
	c[0]=uint32_t(site_index);
	k[0]=uint32_t(site_index);
	RNG::ctr_type r = rng(c, k);
	return delta*(2.0*u01_open_closed_32_53(r[0])-1.0);
}

// standard Gaussian number for (site, time step), Box-Muller on Philox4x32
__device__
float gaussian_noise(int site_index, unsigned long long step, unsigned long noise_seed)
{
	NoiseRNG rng;
	NoiseRNG::ctr_type c={{}};
	NoiseRNG::key_type k={{}};
	c[0]=uint32_t(site_index);
	c[1]=uint32_t(step);
	c[2]=uint32_t(step>>32);
	k[0]=uint32_t(noise_seed);
	k[1]=NOISE_KEY_TAG;
	NoiseRNG::ctr_type r = rng(c, k);
	float u1=u01_open_closed_32_24(r[0]);	// (0,1], so log(u1) is finite
	float u2=u01_closed_open_32_24(r[1]);
	return sqrtf(-2.0f*logf(u1))*cospif(2.0f*u2);
}

__host__ __device__ int index_center(int x,int y,int L){ return x+L*y; }
__host__ __device__ int index_up(int x,int y,int L){ return x+L*((y-1+L)%L); }
__host__ __device__ int index_down(int x,int y,int L){ return x+L*((y+1)%L); }
__host__ __device__ int index_right(int x,int y,int L){ return (x+1)%L+L*y; }
__host__ __device__ int index_left(int x,int y,int L){ return (x-1+L)%L+L*y; }

// deterministic force F_i = c*Laplacian + eps0*[r0*(1+r)*phi - phi^3] + h,
// periodic in x and y
__device__
float deterministic_force(const float *phi, const float *disorder, int L, int i, float h)
{
	int x=i%L;
	int y=i/L;
	float phi_center=phi[i];
	float laplacian = phi[index_up(x,y,L)]+phi[index_down(x,y,L)]
		+phi[index_right(x,y,L)]+phi[index_left(x,y,L)]-4.0f*phi_center;
	float phi4_force = EPSILON0*(R0*(1.0f+disorder[i])*phi_center - phi_center*phi_center*phi_center);
	return CEL*laplacian + phi4_force + h;
}

// one Euler-Maruyama step, phi -> phi_new (Jacobi: reads only phi)
struct eulerop
{
	const float *phi;
	float *phi_new;
	const float *disorder;
	int L;
	float h, dt, noise_amplitude;	// noise_amplitude = sqrt(2*T*dt)
	unsigned long long step;
	unsigned long noise_seed;

	eulerop(const float *_phi, float *_phi_new, const float *_disorder, int _L, float _h, float _dt,
		float _noise_amplitude, unsigned long long _step, unsigned long _noise_seed):
	phi(_phi),phi_new(_phi_new),disorder(_disorder),L(_L),h(_h),dt(_dt),
	noise_amplitude(_noise_amplitude),step(_step),noise_seed(_noise_seed){};

	__device__
	void operator()(int i)
	{
		float value = phi[i] + dt*deterministic_force(phi,disorder,L,i,h);
		if(noise_amplitude>0.0f) value += noise_amplitude*gaussian_noise(i,step,noise_seed);
		phi_new[i]=value;
	}
};

struct disorderop
{
	unsigned long disorder_seed;
	float delta;
	disorderop(unsigned long _disorder_seed, float _delta):disorder_seed(_disorder_seed),delta(_delta){};
	__device__ float operator()(int i){ return site_disorder(i,disorder_seed,delta); }
};

// circular domain of radius R centered at (L/2,L/2): sign_inside inside,
// -sign_inside outside
struct circleop
{
	int L;
	float radius, sign_inside;
	circleop(int _L, float _radius, float _sign_inside):L(_L),radius(_radius),sign_inside(_sign_inside){};
	__host__ __device__ float operator()(int i)
	{
		float dx=float(i%L)-0.5f*L;
		float dy=float(i/L)-0.5f*L;
		return (dx*dx+dy*dy<radius*radius)?sign_inside:-sign_inside;
	}
};

struct is_positive
{
	__host__ __device__
	bool operator()(float x){ return x > 0; }
};

// domain-wall crossings. Each site is first replaced by its 5-point
// (self+4 neighbors) smoothed value, which removes spurious sign flips
// due to thermal noise; a crossing is a sign change of the smoothed
// field along a lattice bond (dir=0: bond to the right neighbor, dir=1:
// bond to the lower neighbor). Checking both bond directions is what
// catches every piece of a closed (e.g. circular) wall. The crossing
// position is linearly interpolated along the bond; bonds without a
// crossing return NaN.
struct wall_crossingop
{
	const float *phi;
	int L, dir;
	wall_crossingop(const float *_phi, int _L, int _dir):phi(_phi),L(_L),dir(_dir){};

	__device__
	float smoothed(int x, int y)
	{
		return phi[index_center(x,y,L)]+phi[index_up(x,y,L)]+phi[index_down(x,y,L)]
			+phi[index_right(x,y,L)]+phi[index_left(x,y,L)];
	}

	__device__
	float2 operator()(int i)
	{
		int x=i%L;
		int y=i/L;
		float s0=smoothed(x,y);
		float s1=(dir==0)?smoothed((x+1)%L,y):smoothed(x,(y+1)%L);
		if(s0*s1>=0.0f) return make_float2(NAN,NAN);
		float frac=s0/(s0-s1);
		if(dir==0){
			float xc=x+frac; if(xc>=L) xc-=L;
			return make_float2(xc,float(y));
		}
		else{
			float yc=y+frac; if(yc>=L) yc-=L;
			return make_float2(float(x),yc);
		}
	}
};

struct is_crossing
{
	__host__ __device__
	bool operator()(float2 p){ return !isnan(p.x); }
};

// (cos, sin) of the x and y angles of positive sites, for the periodic
// (circular-mean) center of mass of the positive domain
struct cm_accum
{
	float cx, sx, cy, sy;
};
struct cm_plus
{
	__host__ __device__
	cm_accum operator()(const cm_accum &a, const cm_accum &b){
		cm_accum r; r.cx=a.cx+b.cx; r.sx=a.sx+b.sx; r.cy=a.cy+b.cy; r.sy=a.sy+b.sy; return r;
	}
};
struct cmop
{
	const float *phi;
	int L;
	cmop(const float *_phi, int _L):phi(_phi),L(_L){};
	__device__
	cm_accum operator()(int i){
		cm_accum a={0,0,0,0};
		if(phi[i]>0){
			float ax=2.0f*float(i%L)/L, ay=2.0f*float(i/L)/L;	// angles in units of pi
			sincospif(ax,&a.sx,&a.cx);
			sincospif(ay,&a.sy,&a.cy);
		}
		return a;
	}
};

// summary of the wall at one time
struct WallStats
{
	int num_wall_points;
	int area_positive, area_negative;
	float magnetization;	// <phi>
	float xcm, ycm;		// periodic center of mass of the positive domain
	float r_mean, r_var;	// mean and variance of the distance wall->cm
};

class System{
	private:
		thrust::device_ptr<float> d_phi;
		thrust::device_ptr<float> d_phi_new;
		thrust::device_ptr<float> d_disorder;
		thrust::device_vector<float2> d_wall;

		int L;
		int L2;

		float h0, freq, temperature, dt;
		unsigned long long step_count;

	public:
		unsigned long disorder_seed, noise_seed;
		float delta;
		float *phi_ptr;
		std::vector<float2> wall;	// host copy of the last detected wall

		System(int _L, float _delta, float _h0, float _freq, float _temperature, float _dt,
			unsigned long _disorder_seed, unsigned long _noise_seed)
		{
			L=_L;
			L2=L*L;
			delta=_delta; h0=_h0; freq=_freq; temperature=_temperature; dt=_dt;
			disorder_seed=_disorder_seed; noise_seed=_noise_seed;
			step_count=0;

			d_phi = thrust::device_malloc<float>(L2);
			d_phi_new = thrust::device_malloc<float>(L2);
			d_disorder = thrust::device_malloc<float>(L2);
			phi_ptr = thrust::raw_pointer_cast(d_phi);

			// the quenched disorder is fixed in time: generate it once
			thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(L2),
				d_disorder, disorderop(disorder_seed,delta));

			thrust::fill(d_phi, d_phi+L2, -1.0f);
		}

		~System(){
			thrust::device_free(d_phi);
			thrust::device_free(d_phi_new);
			thrust::device_free(d_disorder);
		}

		void set_circle(float radius, float sign_inside=1.0f)
		{
			thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(L2),
				d_phi, circleop(L,radius,sign_inside));
		}

		int get_L(){ return L; }
		unsigned long long get_step(){ return step_count; }
		double get_time(){ return double(step_count)*dt; }
		float field(double t){ return h0*cos(2.0*M_PI*freq*t); }
		float get_field(){ return field(get_time()); }

		// num_steps Euler-Maruyama steps; the field is evaluated at the
		// start of each step (explicit/Ito convention)
		void evolve(long num_steps)
		{
			float noise_amplitude=sqrtf(2.0f*temperature*dt);
			float *disorder_ptr=thrust::raw_pointer_cast(d_disorder);
			for(long n=0;n<num_steps;n++)
			{
				thrust::for_each(
					thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(L2),
					eulerop(thrust::raw_pointer_cast(d_phi),thrust::raw_pointer_cast(d_phi_new),
						disorder_ptr,L,get_field(),dt,noise_amplitude,step_count,noise_seed)
				);
				thrust::swap(d_phi,d_phi_new);
				phi_ptr = thrust::raw_pointer_cast(d_phi);
				step_count++;
			}
		}

		float magnetization(){ return thrust::reduce(d_phi,d_phi+L2)/float(L2); }

		// detects the wall (see wall_crossingop), copies it to the host
		// (System::wall) and returns its summary statistics
		WallStats detect_wall()
		{
			WallStats s;
			const float *phi=thrust::raw_pointer_cast(d_phi);
			thrust::counting_iterator<int> first(0), last(L2);

			int n0=thrust::count_if(thrust::make_transform_iterator(first,wall_crossingop(phi,L,0)),
				thrust::make_transform_iterator(last,wall_crossingop(phi,L,0)),is_crossing());
			int n1=thrust::count_if(thrust::make_transform_iterator(first,wall_crossingop(phi,L,1)),
				thrust::make_transform_iterator(last,wall_crossingop(phi,L,1)),is_crossing());
			d_wall.resize(n0+n1);
			thrust::copy_if(thrust::make_transform_iterator(first,wall_crossingop(phi,L,0)),
				thrust::make_transform_iterator(last,wall_crossingop(phi,L,0)),d_wall.begin(),is_crossing());
			thrust::copy_if(thrust::make_transform_iterator(first,wall_crossingop(phi,L,1)),
				thrust::make_transform_iterator(last,wall_crossingop(phi,L,1)),d_wall.begin()+n0,is_crossing());
			wall.resize(n0+n1);
			thrust::copy(d_wall.begin(),d_wall.end(),wall.begin());
			s.num_wall_points=n0+n1;

			s.area_positive=thrust::count_if(thrust::device, phi, phi+L2, is_positive());
			s.area_negative=L2-s.area_positive;
			s.magnetization=magnetization();

			cm_accum zero={0,0,0,0};
			cm_accum a=thrust::transform_reduce(first,last,cmop(phi,L),zero,cm_plus());
			s.xcm=fmod(atan2(a.sx,a.cx)*L/(2*M_PI)+L,double(L));
			s.ycm=fmod(atan2(a.sy,a.cy)*L/(2*M_PI)+L,double(L));

			// distance of each wall point to the cm, minimum-image convention
			double sum_r=0, sum_r2=0;
			for(size_t i=0;i<wall.size();i++){
				float dx=wall[i].x-s.xcm, dy=wall[i].y-s.ycm;
				dx-=L*roundf(dx/L); dy-=L*roundf(dy/L);
				double r=sqrt(dx*dx+dy*dy);
				sum_r+=r; sum_r2+=r*r;
			}
			int n=s.num_wall_points;
			s.r_mean=(n>0)?sum_r/n:0;
			s.r_var=(n>0)?sum_r2/n-s.r_mean*s.r_mean:0;
			return s;
		}

		// writes the last detected wall as one gnuplot "index" block
		void print_wall(std::ofstream &fout, const WallStats &s)
		{
			fout << "# t= " << get_time() << " h= " << get_field() << " n_wall= " << s.num_wall_points
				<< " A+= " << s.area_positive << " A-= " << s.area_negative
				<< " xcm= " << s.xcm << " ycm= " << s.ycm << "\n";
			for(size_t i=0;i<wall.size();i++) fout << wall[i].x << " " << wall[i].y << "\n";
			fout << "\n\n";
		}

		void print_config(std::ofstream &fout)
		{
			thrust::host_vector<float> h_phi(d_phi,d_phi+L2);
			for(int i=0;i<L;i++){
				for(int j=0;j<L-1;j++) fout << h_phi[j+i*L] << " ";
				fout << h_phi[L-1+i*L] << "\n";
			}
			fout << "\n" << std::endl;
		}

		void print_config_linear(std::ofstream &fout)
		{
			thrust::host_vector<float> h_phi(d_phi,d_phi+L2);
			for(int i=0;i<L2;i++) fout << h_phi[i] << "\n";
			fout << "\n" << std::endl;
		}

		void read_config_linear(std::ifstream &fin)
		{
			thrust::host_vector<float> h_phi(L2);
			for(int i=0;i<L2;i++) fin >> h_phi[i];
			thrust::copy(h_phi.begin(),h_phi.end(),d_phi);
		}
};
