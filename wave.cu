////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

#define GLEW_STATIC
#define FREEGLUT_STATIC

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>
using namespace std::chrono;

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
/*
//20/10/19
GLuint vbo2;
struct cudaGraphicsResource *cuda_vbo_resource_2;
void *d_vbo_buffer_2 = NULL;
*/

//22/10/19 test
float4 *h_offsets;
float4 *d_offsets;
//void *offsetsAutoTestH = NULL;
void *offsetsAutoTestD = NULL;

float g_fAnim = 0.0;
float UnitOfChangeOnY = 0.0f;// 19/10/19 test -UI
float UnitOfChangeOnX = 0.0f;

//int jitterAmmountInt = 0;//20/10/19
float jitterAmmountFloat1 = 0.0f;//20/10/19
float jitterAmmountFloat2 = 0.0f;
float jitterAmmountFloat3 = 0.0f;
bool jitter = false;

bool origionalJitter = false;
float jitterAmmountFloatOrigional = 0.0f;
//bool exitTest = false;

//21/10/19 game
//float fallingDistence = 0.0f;
bool falling = false;
float horizontalChange = 0.0f;
float randomHeightTop = 0.0f;
float randomHeightBottom = 0.0f;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);

//20/10/19
void createVBO2(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource, float4 **Hoffsets, float4 **Doffsets);//, float4 * h_offsets, float4 * d_offsets);//, struct cudaGraphicsResource **vbo_resource_2);//20/10/19 test
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time, float UnitOfChangeOnX, float UnitOfChangeOnY, float4 *offsets, bool falling, float horizontalChange, float randomHeightTop, float randomHeightBottom)//, float FallingDistence)//, float jitterAmmountFloat)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	//printf("%d \n", x);
    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
	u = u * 2.0f - 1.0f;//17/10/19 test - making easier to see dots to try make circle (old code - *2.0f - 1.0f;)
	v = v * 2.0f - 1.0f;//*2.0f - 1.0f;
	//16/10/19 test - Q a. start
	if(u > -0.11f + UnitOfChangeOnX & u < 0.11f + UnitOfChangeOnX){// u > -0.11f & u < 0.11f (new reduced x values)
		if(v > -0.125f + UnitOfChangeOnY & v < 0.125f + UnitOfChangeOnY){
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);
		}
		else {
			float freq = 4.0f;
			float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

			if (falling) {
				if(u > 0.75f & u < 1.0f){//bottom block
					if(v > 0.90f - randomHeightBottom & v < 1.0f){
						//printf("in if");
						//float gameTime = time;
						pos[y*width + x] = make_float4(u-horizontalChange, 0.0f, v, 1.0f);
					}
					else {
						pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
					}
				}
				else {
					pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
				}
				
				if (u > 0.75f & u < 1.0f) //{//top block
					if (v > -1.1f & v < -0.90f + randomHeightTop) //{
						//printf("in if");
						//float gameTime = time;
						pos[y*width + x] = make_float4(u - horizontalChange, 0.0f, v, 1.0f);
					//}
					//else {
					//	pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
					//}
				//}
				//else {
				//	pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
				//}
				
			}
			else {
				// write output vertex
				pos[y*width + x] = make_float4(u + offsets[y*width + x].x, w + offsets[y*width + x].y, v + offsets[y*width + x].z, 1.0f);
			}
		}
	}
	else {
		//printf("*** IN ELSE *** \n");
		// calculate simple sine wave pattern
		float freq = 4.0f;
		float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
		if(falling){
			if(u > 0.75f & u < 1.0f){//bottom block
				if(v > 0.90f - randomHeightBottom & v < 1.0f){
					//printf("in if");
					//float gameTime = time;
					pos[y*width + x] = make_float4(u-horizontalChange, 0.0f, v, 1.0f);
				}
				else {
					pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
				}
			}
			else {
				pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
			}
			
			if (u > 0.75f & u < 1.0f) //{//top block
				if (v > -1.1f & v < -0.90f + randomHeightTop) //{
					//printf("in if");
					//float gameTime = time;
					pos[y*width + x] = make_float4(u - horizontalChange, 0.0f, v, 1.0f);
				//}
				//else {
				//	pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
				//}
			//}
			//else {
			//	pos[y*width + x] = make_float4(u, -0.5f, v, 1.0f);
			//}
			
		}
		else {
			// write output vertex
			pos[y*width + x] = make_float4(u + offsets[y*width + x].x, w + offsets[y*width + x].y, v + offsets[y*width + x].z, 1.0f);
		}
	}
	//16/10/19 test - Q a. end
	__syncthreads();
	//17/10/19 test - Q a. start
	//__syncthreads();//extra top part
	//if (u > -0.109f & u < 0.109f)
	//	if (v > -0.126f & v < 0.126f)
	//		pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);
	
	if (u > -0.111f + UnitOfChangeOnX & u < 0.111f + UnitOfChangeOnX)//1 - u > -0.111f & u < 0.111f (new reduced x values)
		if (v > -0.109f + UnitOfChangeOnY & v < 0.109f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);
	//__syncthreads();
	//if (u > -0.126f & u < 0.126f)//test
	//	if (v > -0.125f & v < 0.125f)
	//		pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	__syncthreads();//2
	if (u > -0.127f + UnitOfChangeOnX & u < 0.127f + UnitOfChangeOnX)// u > -0.127f & u < 0.127f (new reduced x values)
		if (v > -0.093f + UnitOfChangeOnY & v < 0.093f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	__syncthreads();//3
	if (u > -0.143f + UnitOfChangeOnX & u < 0.143f + UnitOfChangeOnX)// u > -0.143f & u < 0.143f (new reduced x values)
		if (v > -0.077f + UnitOfChangeOnY & v < 0.077f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	__syncthreads();//4
	if (u > -0.148f + UnitOfChangeOnX & u < 0.148f + UnitOfChangeOnX)// u > -0.148f & u < 0.148f (new reduced x values)
		if (v > -0.061f + UnitOfChangeOnY & v < 0.061f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	__syncthreads();//5
	if (u > -0.164f + UnitOfChangeOnX & u < 0.164f + UnitOfChangeOnX)// u > -0.164f & u < 0.164f (new reduced x values)
		if (v > -0.045f + UnitOfChangeOnY & v < 0.045f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);
	
	__syncthreads();// "top part" 1
	if (u > -0.094f + UnitOfChangeOnX & u < 0.094f + UnitOfChangeOnX)
		if (v > -0.141f + UnitOfChangeOnY & v < 0.141f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	__syncthreads();// "top part" 2
	if (u > -0.078f + UnitOfChangeOnX & u < 0.078f + UnitOfChangeOnX)
		if (v > -0.157f + UnitOfChangeOnY & v < 0.157f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	__syncthreads();// "top part" 3
	if (u > -0.062f + UnitOfChangeOnX & u < 0.062f + UnitOfChangeOnX)
		if (v > -0.173f + UnitOfChangeOnY& v < 0.173f + UnitOfChangeOnY)
			pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);
	
	//17/10/19 test - Q a. end
}

__global__ void new_vbo_x_kernel(float4 *pos, unsigned int width, unsigned int height, float time, float jitterAmmountFloatOrigional)//, float4 *randNum)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	//printf("JitterAmmount: %f\n", jitterAmmountFloat);
	//printf("")
	/*
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	//pos[y*width + x] = make_float4(u + jitterAmmountFloat, 0.0f, v + jitterAmmountFloat, 1.0f);
	*/

	//pos[y*width + x].x = u + jitterAmmountFloat;//make_float4(u + jitterAmmountFloat, 0.0f + jitterAmmountFloat, v + jitterAmmountFloat, 1.0f);
	pos[y*width + x].x += pos[y*width + x].x * jitterAmmountFloatOrigional;
	//pos[y*width + x].y += pos[y*width + x].y * jitterAmmountFloat;
	//pos[y*width + x].z += pos[y*width + x].z * jitterAmmountFloat;
	
}

__global__ void new_vbo_y_kernel(float4 *pos, unsigned int width, unsigned int height, float time, float jitterAmmountFloatOrigional)//, float4 *randNum)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	/*
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	//pos[y*width + x].y = u + jitterAmmountFloat;
	*/
	/*
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
	pos[y*width + x].y = w + jitterAmmountFloat;//make_float4(u, w + jitterAmmountFloat, v, 1.0f);
	*/
	
	pos[y*width + x].y += pos[y*width + x].y * jitterAmmountFloatOrigional;
}

__global__ void new_vbo_z_kernel(float4 *pos, unsigned int width, unsigned int height, float time, float jitterAmmountFloatOrigional)//, float4 *randNum)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	/*
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	pos[y*width + x].z = v + jitterAmmountFloat;
	*/
	pos[y*width + x].z += pos[y*width + x].z * jitterAmmountFloatOrigional;
}

__global__ void game_kernel(float4 *pos, unsigned int width, unsigned int height, float time, bool falling, float UnitOfChangeOnX, float UnitOfChangeOnY)//, float jitterAmmountFloatOrigional)//, float seedTest)//, float4 *randNum)
{
	/*
	if (falling) {
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		float u = x / (float)width;
		float v = y / (float)height;
		u = u * 2.0f - 1.0f;
		v = v * 2.0f - 1.0f;

		if (u > 0.0f & u < 1.0f)
			if (v > 0.0f & v < -1.0f)
				pos[y*width + x] = make_float4(u, 0.0f, v, 1.0f);

	}
	*/
}

void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time, float4 *offsets)//, float4 *randNum)
{
    // execute the kernel
	
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);

	auto seedTest = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::mt19937 generatorTest;
	generatorTest.seed(seedTest);
	std::uniform_real_distribution<double> distributionTest(0.0, 0.75);
	auto seedTest2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::mt19937 generatorTest2;
	generatorTest2.seed(seedTest2);

	if (falling)
		UnitOfChangeOnY += 0.016;
	if (falling & UnitOfChangeOnY > 1.0f)
		falling = false;//UnitOfChangeOnY = 0.0f;
	if (falling)
		horizontalChange += 0.016f;
	if (falling & horizontalChange > 2.0f)
		horizontalChange = 0.0f;
	if (horizontalChange == 0.0f)
		randomHeightTop = distributionTest(generatorTest);
	if (horizontalChange == 0.0f)
		randomHeightBottom = distributionTest(generatorTest2);
	//auto seedTest = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	game_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time, falling, UnitOfChangeOnX, UnitOfChangeOnY);

	simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time, UnitOfChangeOnX, UnitOfChangeOnY, offsets,falling, horizontalChange, randomHeightTop, randomHeightBottom);//, fallingDistence);//, jitterAmmountFloat);
	//20/10/19 test - jitter
	//if (exitTest != true) {
	    
		if (origionalJitter) {
			//srand(static_cast<unsigned int>(clock()));
			//jitterAmmountFloat = float(rand()) / (float(RAND_MAX)  * 2.0f - 1.0f);//+ 1.0);

			//std::default_random_engine generator;
			//milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
			//float testFloat = (float)ms;

			auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			std::mt19937 generator;
			generator.seed(seed);//std::time(0));// * 1000);
			std::uniform_real_distribution<double> distribution(-0.0625 , 0.0652);//-0.03125f, 0.03125f); //doubles from 
			jitterAmmountFloatOrigional = distribution(generator);
			//printf("JitterAmmount x: %f\n", jitterAmmountFloatOrigional);
			new_vbo_x_kernel << <grid, block >> > (pos, mesh_width, mesh_height, time, jitterAmmountFloatOrigional);//, randNum);

			auto seed2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			generator.seed(seed2);//std::time(0));// * 1000); 
			jitterAmmountFloatOrigional = distribution(generator);
			//printf("JitterAmmount y: %f\n", jitterAmmountFloatOrigional);
			new_vbo_y_kernel << <grid, block >> > (pos, mesh_width, mesh_height, time, jitterAmmountFloatOrigional);

			auto seed3 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			generator.seed(seed3);//std::time(0));// * 1000); 
			jitterAmmountFloatOrigional = distribution(generator);
			//printf("JitterAmmount z: %f\n", jitterAmmountFloatOrigional);
			new_vbo_z_kernel << <grid, block >> > (pos, mesh_width, mesh_height, time, jitterAmmountFloatOrigional);
		}
		
	//}
}

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

    printf("\n");

    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    // command line mode only
    if (ref_file != NULL)
    {
        // create VBO
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

		//22/10/19 - start
		unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);//1.
		//float4 *h_offsets;//2. MIGHT HAVE TO MAKE THESE GLOBALS
		//float4 *d_offsets;// MIGHT HAVE TO MAKE THESE GLOBALS
		//float4 **testh = &h_offsets;
		//float4 **testd = &d_offsets;
		h_offsets = (float4*)malloc(size);//test - size);//3
		//cudaMalloc(&d_offsets, size);//test - size);
		cudaMalloc((void **)&d_offsets, size);//exactly lab3 way

		//for (int i = 0; i < size; i++) { h_offsets[i] = { 0.1f, 0.1f, 0.1f, 0.1f }; }//4 - fill h_offset with random xyz THESE MIGHT HAVE TO HAPPEN IN RUNCUDA

		//cudaMemcpy(d_offsets, h_offsets, size, cudaMemcpyHostToDevice);//5 THESE MIGHT HAVE TO HAPPEN IN RUNCUDA
		/*
		if (jitter) {
			for (int i = 0; i < size; i++) { h_offsets[i] = make_float4(0.5f, 0.5f, 0.5f, 1.0f); }//4
		}
		else {
			for (int i = 0; i < 20000; i++) { h_offsets[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f); }//4
		}
		cudaMemcpy(d_offsets, h_offsets, size, cudaMemcpyHostToDevice);//5
		*/
		//22/10/19 - end

		//20/10/19
		//createVBO2(&vbo2, &cuda_vbo_resource_2, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
		runCuda(&cuda_vbo_resource, &h_offsets, &d_offsets);//, &cuda_vbo_resource_2);//, 0);// 18/10/19 test - UI

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource, float4 **Hoffsets, float4 **Doffsets)//, struct cudaGraphicsResource **vbo_resource_2)// 18/10/19 test - UI
{
	
		// map OpenGL buffer object for writing from CUDA
		float4 *dptr;
		checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
			*vbo_resource));
		//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

		// execute the kernel
		//    dim3 block(8, 8, 1);
		//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
		//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

		/*
		//20/10/19
		float4 *dptr2;
		checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource_2, 0));
		size_t num_bytes_2;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr2, &num_bytes_2,
			*vbo_resource_2));
		*/
		//22/10/19 test - start
		
		float4 *dptr2;//23/10/19
		float4 *dptr3;//23/10/19
		dptr2 = *Hoffsets;
		dptr3 = *Doffsets;
		unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
		if (jitter) {
			auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			std::mt19937 generator;
			generator.seed(seed);
			std::uniform_real_distribution<double> distribution(-0.0625, 0.0652);
			jitterAmmountFloat1 = distribution(generator);
			//printf("JitterAmmountFloat1: %f\n", jitterAmmountFloat1);

			auto seed2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			generator.seed(seed2);
			jitterAmmountFloat2 = distribution(generator);
			//printf("JitterAmmountFloat2: %f\n", jitterAmmountFloat2);

			auto seed3 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			generator.seed(seed3);
			jitterAmmountFloat3 = distribution(generator);
			//printf("JitterAmmountFloat3: %f\n", jitterAmmountFloat3);
			for (int i = 0; i < 60000; i++) { dptr2[i] = make_float4(jitterAmmountFloat1, jitterAmmountFloat2, jitterAmmountFloat3, 1.0f); }//4
		}
		else {
			for (int i = 0; i < 60000; i++) { dptr2[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f); }//4
		}
		cudaMemcpy(dptr3, dptr2, 60000, cudaMemcpyHostToDevice);//5
		
		//22/10/19 test - end
		launch_kernel(dptr, mesh_width, mesh_height, g_fAnim, dptr3);//, d_offsets);//, dptr2);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
	//checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource_2, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(mesh_width*mesh_height*sizeof(float));
	//char ui = ' '; 18/10/19 test - UI
    // execute the kernel
	launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim, (float4 *)offsetsAutoTestD);//, d_offsets);//, (float4 *)d_vbo_buffer_2);//, 0);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                mesh_width*mesh_height*sizeof(float),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}
/*
void createVBO2(GLuint *vbo2, struct cudaGraphicsResource **vbo_res_2,
	unsigned int vbo_res_flags_2)
{
	assert(vbo2);

	// create buffer object
	glGenBuffers(1, vbo2);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo2);

	// initialize buffer object
	unsigned int size = ((float(rand()) / float(RAND_MAX)) * (1.0f - -1.0f)) + -1.0f;
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 1);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res_2, *vbo2, vbo_res_flags_2));

	SDK_CHECK_ERROR_GL();
}
*/
////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);
    // run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource, &h_offsets, &d_offsets);//, h_offsets, d_offsets);//, &cuda_vbo_resource_2);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
		//deleteVBO(&vbo2, cuda_vbo_resource_2);//20/10/19
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	//float4 * dptr;//18/10/19 test - UI
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif

		//18/10/19 test - UI
		case (115) :
			//launch_new_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim, 1);
			//runCuda(&cuda_vbo_resource, 1);
			if(!falling)
				if(UnitOfChangeOnY < 1.072f)
					UnitOfChangeOnY += 0.032f;// 19/10/19 test - UI
			//display();// 19/10/19 test - UI
			printf("JUST_PRESSED_s \n");
			return;

		case (119):
			if (UnitOfChangeOnY > -1.072f)
				UnitOfChangeOnY -= 0.032f;

			if(falling)
				if(UnitOfChangeOnY > -1.072f)
					UnitOfChangeOnY -= 0.256f;

			printf("JUST_PRESSED_w \n");
			return;

		case (97):
			if (!falling)
				if (UnitOfChangeOnX > -1.072f)
					UnitOfChangeOnX -= 0.032f;
			printf("JUST_PRESSED_a \n");
			return;

		case (100):
			if (!falling)
				if (UnitOfChangeOnX < 1.072f)
					UnitOfChangeOnX += 0.032f;
			printf("JUST_PRESSED_d \n");
			return;

		case (49):
			//if (exitTest)
			//	exitTest = false;
			jitter = true;
			//srand(static_cast<unsigned int>(clock()));
			//jitterAmmountInt = 0 + (rand() %2);//(rand() / (float)RAND_MAX * 1.0f) + -1.0f;//((float(rand()) / float(RAND_MAX)) * 1.0f - -1.0f);//(1.0f - -1.0f)) + -1.0f;//* (1.0f - -1.0f)) + -1.0f;
			//jitterAmmountFloat = double(rand()) / (double(RAND_MAX) + 1.0);//float(5 + rand() % (150 +1 -5))/ 100;
			//printf("JitterAmmountInt: %d\n", jitterAmmountInt);
			//printf("JitterAmmountFloat: %f\n", jitterAmmountFloat);
			printf("JUST_PRESSED_1 \n");
			return;

		case (50):
			//exitTest = true;
			jitter = false;
			printf("JUST_PRESSED_2 \n");
			return;

		case (53):
			falling = true;
			UnitOfChangeOnX = 0.0f;
			UnitOfChangeOnY = 0.0f;
			/*
			while (falling)
			{
				if (fallingDistence > 1.0f)
					fallingDistence = 0.0f;

				fallingDistence += 0.016f;
			}
			*/
			printf("JUST_PRESSED_5 \n");
			return;

		case(54):
			falling = false;
			//fallingDistence = 0.0f;
			printf("JUST_PRESSED_6 \n");
			return;

		case(51):
			origionalJitter = true;
			printf("JUST_PRESSED_3 \n");
			return;

		case(52):
			origionalJitter = false;
			printf("JUST_PRESSED_4 \n");
			return;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
