// nvcc -Xcompiler -Wall -DDOLOG ppm-cuda.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>

// https://stackoverflow.com/questions/28896001/read-write-to-ppm-image-file-c

#define DO_READ

#ifdef DOLOG
#define LOG(msg) std::cerr<<msg<<std::endl
//#define LOG(msg) fprintf(stderr, msg "\n");
#else
#define LOG(msg)
#endif

// host code for validating last cuda operation (not kernel launch)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



char* data;

int read(std::string filename,
         int& width,
         int& height,
         std::vector<float>& r,
         std::vector<float>& g,
         std::vector<float>& b)
{
    std::ifstream in(filename.c_str(), std::ios::binary);

    int maxcol;

    if (! in.is_open())
    {
        std::cerr << "could not open " << filename << " for reading" << std::endl;
        return 0;
    }

    {
        std::string magicNum;
        in >> magicNum;
        LOG("got magicNum:" << magicNum);

        // this is broken if magicNum != 'P6'
    }

    {
        long loc = in.tellg();
        std::string comment;
        in >> comment;

        if (comment[0] != '#')
        {
            in.seekg(loc);
        }
        else
        {
            LOG("got comment:" << comment);
        }
    }

    in >> width >> height >> maxcol;
    in.get();                   // eat newline
    LOG("dimensions: " << width << "x" << height << "("<<maxcol<<")");
    

//    char* data = new char[width*height*3];
    data = new char[width*height*3];
    in.read(data, width*height*3);
    in.close();
    
    r.resize(width*height);
    g.resize(width*height);
    b.resize(width*height);

    for (int i=0; i<width*height; ++i)
    {
        int base = i*3;
        r[i] =  ((unsigned char)data[base+0])/255.0f;
        g[i] =  ((unsigned char)data[base+1])/255.0f;
        b[i] =  ((unsigned char)data[base+2])/255.0f;
    }
    free(data);

    return 1;
}


int write(std::string outfile,
          int width, int height,
          const std::vector<float>& r,
          const std::vector<float>& g,
          const std::vector<float>& b)
{
    std::ofstream ofs(outfile.c_str(), std::ios::out | std::ios::binary);

    if (! ofs.is_open())
    {
        std::cerr << "could not open " << outfile << " for writing" << std::endl;
    }

    ofs << "P6\n#*\n" << width << " " << height << "\n255\n";

    for (int i=0; i < width*height; ++i)
    {
        ofs <<
            (unsigned char)(r[i]*255) <<
            (unsigned char)(g[i]*255) <<
            (unsigned char)(b[i]*255);
    }
    ofs.close();
    
    return 1;
}



#define imin(a,b) (a<b?a:b)

__global__ void edgeDetect(int width, int height, float* r, float* g, float* b, float* rnew, float* gnew, float* bnew, float delta)
{
    // thread's .x coordinates are the pixel column in the image
    // thread's .y coordinates are the pixel row in the image

    int global_pixel_row=threadIdx.y + blockIdx.y*blockDim.y;
    int global_pixel_col=threadIdx.x + blockIdx.x*blockDim.x;

    if (global_pixel_col < width &&  global_pixel_row < height)
    {
        // this pixel exists in the image and this is not an idle thread

        // find the index in r,g,b for this pixel
        int index=global_pixel_col + global_pixel_row * width;
	int rightIdx = 0;
	int leftIdx = 0;
	int topIdx = 0;
	int botIdx = 0;	
	float dx;
	float dy;
        // Perceived Intensities
	float Iright;
	float Ileft;
	float Itop;
	float Ibot;
	float gradD;

        // Only blur interior pixels
        if (global_pixel_row > 1 && global_pixel_row < height && 
	    global_pixel_col > 1 && global_pixel_col < width)
        {
	    rightIdx = index + 1;
	    leftIdx = index - 1;
	    topIdx = global_pixel_col + (global_pixel_row - 1) * width;
   	    botIdx = global_pixel_col + (global_pixel_row + 1) * width;
            
	    // Compute Perceived Intensity at each neighbor        
	    Iright = 0.2126 * r[rightIdx] + 0.7152 * g[rightIdx] + 0.0722 * b[rightIdx];
	    Ileft = 0.2126 * r[leftIdx] + 0.7152 * g[leftIdx] + 0.0722 * b[leftIdx];
	    Itop = 0.2126 * r[topIdx] + 0.7152 * g[topIdx] + 0.0722 * b[topIdx];
	    Ibot = 0.2126 * r[botIdx] + 0.7152 * g[botIdx] + 0.0722 * b[botIdx];

	    dx = -0.5 * Ileft + 0.5 * Iright;
	    dy = -0.5 * Ibot + 0.5 * Itop;             
            
            gradD = sqrt((dx * dx) + (dy * dy));

	    if (gradD >= delta) 
	    {
		rnew[index] = 1;
		gnew[index] = 1;
		bnew[index] = 1;
	    }
	    else 
	    {
		rnew[index] = 0;
		gnew[index] = 0;
		bnew[index] = 0;
	    }
	
        }   
    }
    else
    {
        // do nothing
    }





    // image data (r,g,b) is stored row-major (all of pixel row 0, followed by all of pixel row 1, etc.)
    

    

}



int main(int argc, char *argv[])
{
    int width, height;

    std::vector<float> r,g,b, rnew, gnew, bnew;
    float *d_r, *d_g, *d_b, *d_rnew, *d_gnew, *d_bnew;

#ifdef DO_READ

    read("xwing.ppm", width, height, r,g,b);
    LOG("processing " << width << "x" << height);

#else

    width=640;
    height=480;

    gpuErrchk(cudaMalloc(&d_r, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_g, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, width*height*sizeof(float)));

#endif
    

    
    // call kernel

    dim3 tpb(32, 32);
    //dim3 tpb(1, 1);
    dim3 bpg((width+tpb.x-1)/tpb.x, (height+tpb.y-1)/tpb.y);

    
    float delta = 0.2;

    // Load Image to GPU
    gpuErrchk(cudaMalloc(&d_r, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_g, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, width*height*sizeof(float)));

    gpuErrchk(cudaMalloc(&d_rnew, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_gnew, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_bnew, width*height*sizeof(float)));

    gpuErrchk(cudaMemcpy(d_r, &r[0], width*height*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_g, &g[0], width*height*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, &b[0], width*height*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    
    edgeDetect<<<bpg,tpb>>>(width, height, d_r, d_g, d_b, d_rnew, d_gnew, d_bnew, delta);

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("cuda kernel: %.2f ms\n", elapsed);


    // check to see if there were any issues with the previous kernel launch
    gpuErrchk( cudaPeekAtLastError() );

    // copy new back from GPU
    rnew.resize(width*height);
    gnew.resize(width*height);
    bnew.resize(width*height);
       
    gpuErrchk(cudaMemcpy(&rnew[0], d_rnew, width*height*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&gnew[0], d_gnew, width*height*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&bnew[0], d_bnew, width*height*sizeof(float), cudaMemcpyDeviceToHost));




    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_rnew);
    cudaFree(d_gnew);
    cudaFree(d_bnew);   
    // save image
    write("edges.ppm", width, height, rnew,gnew,bnew);

    return 0;
}
