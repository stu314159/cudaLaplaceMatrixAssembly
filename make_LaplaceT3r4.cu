#include <iostream>
#include <ctime>
#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include <cusp/print.h>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>

using namespace std;

const int BLOCK_SZ=128; //number of elements to be processed in a given block

typedef float Real;
typedef int cInt;

//--- general functions and functors to be used.
//implies a row-major ordering of the elements.
struct key_compute_functor
{
  const int num_cols;
  key_compute_functor(int _cols) : num_cols(_cols){}

  __host__ __device__
  int operator()(const int& i, const int& j)const{
    return (i*num_cols+j);
  }
};

void compute_keys(int num_cols, thrust::device_vector<int>& I,
		  thrust::device_vector<int>& J){

  thrust::transform(I.begin(),I.end(),J.begin(),J.begin(),
		    key_compute_functor(num_cols));

}

// need to do the opposite...get the column index from key value
struct colInd_compute_functor
{
  const int num_cols;
  colInd_compute_functor(int _cols) : num_cols(_cols){}
  __host__ __device__
  int operator()(const int& key_val) const {
    return(key_val%num_cols);
  }

};

void compute_colInd(int num_cols,int nnz,thrust::device_vector<int>& key,
		    thrust::device_vector<int>& J){

  thrust::transform(key.begin(),key.begin()+nnz,J.begin(),
		    colInd_compute_functor(num_cols));
}

struct rowInd_compute_functor
{
  const int num_cols;
  rowInd_compute_functor(int _cols) : num_cols(_cols){}
  __host__ __device__
  int operator()(const int& key_val) const{
    return(key_val/num_cols);
  }

};

void compute_rowInd(int num_cols,int nnz,thrust::device_vector<int>& key,
		    thrust::device_vector<int>& I){
  thrust::transform(key.begin(),key.begin()+nnz,I.begin(),
		    rowInd_compute_functor(num_cols));
}


template < class T >
void print_thrust(thrust::device_vector<T>& V){
  thrust::copy(V.begin(),V.end(),std::ostream_iterator<T>(std::cout, "\n"));
}


template < class T >
int sort_and_combine(thrust::device_vector<int>& I, thrust::device_vector<int>& J,
		     thrust::device_vector<T>& V, int num_triplets,
		     int num_rows, int num_cols){
 
  // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
  thrust::stable_sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
  thrust::stable_sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));

  int nnz = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
				  thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
				  thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
				  int(0),
				  thrust::plus<int>(),
				  thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

  //int nnz = new_end.first - key_d2.begin(); 

  // allocate output matrix
  cusp::coo_matrix<int, T, cusp::device_memory> A(num_rows, num_cols, nnz);

  // sum values with the same (i,j) index
  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
			V.begin(),
			thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
			A.values.begin(),
			thrust::equal_to< thrust::tuple<int,int> >(),
			thrust::plus<float>());


  thrust::copy(A.row_indices.begin(),A.row_indices.end(),I.begin());
  thrust::copy(A.column_indices.begin(),A.column_indices.end(),J.begin());
  thrust::copy(A.values.begin(),A.values.end(),V.begin());
  // I = A.row_indices;
  // J = A.column_indices;
  // V = A.values;

  return nnz;

}

template < class T >
void copy_to_csr(cusp::csr_matrix<int,T,cusp::device_memory>& L_csr,
		 const thrust::device_vector<int>& I, 
		 const thrust::device_vector<int>& J,
		 const thrust::device_vector<T>& S,
		 int nrow, int ncol,int nnz){
  //make temporaries to hold the non-zero data
  thrust::device_vector<int> Iv(nnz);
  thrust::device_vector<int> Jv(nnz);
  thrust::device_vector<T> Sv(nnz);

  //copy non-zero data over to the temporaries
  thrust::copy(I.begin(),I.begin()+nnz,Iv.begin());
  thrust::copy(J.begin(),J.begin()+nnz,Jv.begin());
  thrust::copy(S.begin(),S.begin()+nnz,Sv.begin());

  //make a temporary COO matrix to hold the non-zero data
  cusp::coo_matrix<int,T,cusp::device_memory> L_coo(nrow,ncol,nnz);
  L_coo.row_indices = Iv;
  L_coo.column_indices = Jv;
  L_coo.values = Sv;

  //copy the COO data to CSR form to return.
  L_csr = L_coo;//finally, copy the data over to the CSR matrix

}


//--- kernel to generate vector values ----

__global__ void compute_vectors(int * i_v, int * j_v, Real * s_v,
				Real * gcoord_d, int * nodes_d,
				int nnodes, int nel){

  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nel){

    //since each element will have 27 entries, start my 
    //entry index appropriately.
    int entry_index = tid*27;

    //get the global node numbers of my 3 local nodes
    int nd1=nodes_d[tid];
    int nd2=nodes_d[nel+tid];
    int nd3=nodes_d[2*nel+tid];
    //since this is a scalar problem, take advantage of the fact
    //that the global dof = global node number.


    //convert to make nd1,nd2 and nd3 zero-based...
    //to make 1-based again, delete this line...
    nd1-=1; nd2-=1;nd3-=1;

    //get the x and y coordinates of my nodes.

    //to make ones-based again, subtract 1 from these indices...
    Real x1 = gcoord_d[nd1];
    Real x2 = gcoord_d[nd2];
    Real x3 = gcoord_d[nd3];

    Real y1 = gcoord_d[nnodes+nd1];
    Real y2 = gcoord_d[nnodes+nd2];
    Real y3 = gcoord_d[nnodes+nd3];

    //turns out, these values aren't even used for the Laplacian...
    // // first integration point: xi = 1./6., eta = 1./6., wt = 1./6.
    // Real xi = 1./6.;
    // Real eta = 1./6.;
    Real wt = 1./6.;

    // //get the values of the shape functions, and derivatives for this
    // //gauss point.
    // Real shape1=1.-xi-eta;
    // Real shape2 = xi;
    // Real shape3 = eta;

    //note, these are constants...
    Real dhdr1=-1.;
    Real dhdr2 = 1.;
    Real dhdr3 = 0.;

    Real dhds1 = -1.;
    Real dhds2 = 0.;
    Real dhds3 = 1.;

    //compute the jacobian
    Real jac11, jac12, jac21, jac22;
    jac11 = 0; jac12 = 0; jac21 = 0; jac22 = 0;
    jac11=dhdr1*x1+dhdr2*x2+dhdr3*x3;
    jac12=dhdr1*y1+dhdr2*y2+dhdr3*y3;
    jac21=dhds1*x1+dhds2*x2+dhds3*x3;
    jac22=dhds1*y1+dhds2*y2+dhds3*y3;


    Real det_jac;
    det_jac = jac11*jac22-jac12*jac21;
    
    Real inv_jac11, inv_jac12, inv_jac21, inv_jac22;
    inv_jac11=(1./det_jac)*jac22;
    inv_jac12=-(1./det_jac)*jac12;
    inv_jac21=-(1./det_jac)*jac21;
    inv_jac22=(1./det_jac)*jac11;

    Real dhdx1,dhdx2,dhdx3,dhdy1,dhdy2,dhdy3;

    dhdx1=inv_jac11*dhdr1+inv_jac12*dhds1;
    dhdx2=inv_jac11*dhdr2+inv_jac12*dhds2;
    dhdx3=inv_jac11*dhdr3+inv_jac12*dhds3;

    dhdy1=inv_jac21*dhdr1+inv_jac22*dhds1;
    dhdy2=inv_jac21*dhdr2+inv_jac22*dhds2;
    dhdy3=inv_jac21*dhdr3+inv_jac22*dhds3;

    //the only thing that changes between integration points are xi and
    //eta and these only affect the value of the shape functions....
    //but these are not used for the Laplacian, so there is no point
    //in re-computing this for each integration point...

    i_v[entry_index]=nd1;
    j_v[entry_index]=nd1;
    s_v[entry_index]=(dhdx1*dhdx1+dhdy1*dhdy1)*wt*det_jac;

    i_v[entry_index+1]=nd1;
    j_v[entry_index+1]=nd2;
    s_v[entry_index+1]=(dhdx1*dhdx2+dhdy1*dhdy2)*wt*det_jac;

    i_v[entry_index+2]=nd1;
    j_v[entry_index+2]=nd3;
    s_v[entry_index+2]=(dhdx1*dhdx3+dhdy1*dhdy3)*wt*det_jac;

    i_v[entry_index+3]=nd2;
    j_v[entry_index+3]=nd1;
    s_v[entry_index+3]=(dhdx2*dhdx1+dhdy2*dhdy1)*wt*det_jac;

    i_v[entry_index+4]=nd2;
    j_v[entry_index+4]=nd2;
    s_v[entry_index+4]=(dhdx2*dhdx2+dhdy2*dhdy2)*wt*det_jac;

    i_v[entry_index+5]=nd2;
    j_v[entry_index+5]=nd3;
    s_v[entry_index+5]=(dhdx2*dhdx3+dhdy2*dhdy3)*wt*det_jac;

    i_v[entry_index+6]=nd3;
    j_v[entry_index+6]=nd1;
    s_v[entry_index+6]=(dhdx3*dhdx1+dhdy3*dhdy1)*wt*det_jac;

    i_v[entry_index+7]=nd3;
    j_v[entry_index+7]=nd2;
    s_v[entry_index+7]=(dhdx3*dhdx2+dhdy3*dhdy2)*wt*det_jac;

    i_v[entry_index+8]=nd3;
    j_v[entry_index+8]=nd3;
    s_v[entry_index+8]=(dhdx3*dhdx3+dhdy3*dhdy3)*wt*det_jac;

    // second gauss point

    i_v[entry_index+9]=nd1;
    j_v[entry_index+9]=nd1;
    s_v[entry_index+9]=(dhdx1*dhdx1+dhdy1*dhdy1)*wt*det_jac;

    i_v[entry_index+10]=nd1;
    j_v[entry_index+10]=nd2;
    s_v[entry_index+10]=(dhdx1*dhdx2+dhdy1*dhdy2)*wt*det_jac;

    i_v[entry_index+11]=nd1;
    j_v[entry_index+11]=nd3;
    s_v[entry_index+11]=(dhdx1*dhdx3+dhdy1*dhdy3)*wt*det_jac;

    i_v[entry_index+12]=nd2;
    j_v[entry_index+12]=nd1;
    s_v[entry_index+12]=(dhdx2*dhdx1+dhdy2*dhdy1)*wt*det_jac;

    i_v[entry_index+13]=nd2;
    j_v[entry_index+13]=nd2;
    s_v[entry_index+13]=(dhdx2*dhdx2+dhdy2*dhdy2)*wt*det_jac;

    i_v[entry_index+14]=nd2;
    j_v[entry_index+14]=nd3;
    s_v[entry_index+14]=(dhdx2*dhdx3+dhdy2*dhdy3)*wt*det_jac;

    i_v[entry_index+15]=nd3;
    j_v[entry_index+15]=nd1;
    s_v[entry_index+15]=(dhdx3*dhdx1+dhdy3*dhdy1)*wt*det_jac;

    i_v[entry_index+16]=nd3;
    j_v[entry_index+16]=nd2;
    s_v[entry_index+16]=(dhdx3*dhdx2+dhdy3*dhdy2)*wt*det_jac;

    i_v[entry_index+17]=nd3;
    j_v[entry_index+17]=nd3;
    s_v[entry_index+17]=(dhdx3*dhdx3+dhdy3*dhdy3)*wt*det_jac;

    //third gauss point

    i_v[entry_index+18]=nd1;
    j_v[entry_index+18]=nd1;
    s_v[entry_index+18]=(dhdx1*dhdx1+dhdy1*dhdy1)*wt*det_jac;

    i_v[entry_index+19]=nd1;
    j_v[entry_index+19]=nd2;
    s_v[entry_index+19]=(dhdx1*dhdx2+dhdy1*dhdy2)*wt*det_jac;

    i_v[entry_index+20]=nd1;
    j_v[entry_index+20]=nd3;
    s_v[entry_index+20]=(dhdx1*dhdx3+dhdy1*dhdy3)*wt*det_jac;

    i_v[entry_index+21]=nd2;
    j_v[entry_index+21]=nd1;
    s_v[entry_index+21]=(dhdx2*dhdx1+dhdy2*dhdy1)*wt*det_jac;

    i_v[entry_index+22]=nd2;
    j_v[entry_index+22]=nd2;
    s_v[entry_index+22]=(dhdx2*dhdx2+dhdy2*dhdy2)*wt*det_jac;

    i_v[entry_index+23]=nd2;
    j_v[entry_index+23]=nd3;
    s_v[entry_index+23]=(dhdx2*dhdx3+dhdy2*dhdy3)*wt*det_jac;

    i_v[entry_index+24]=nd3;
    j_v[entry_index+24]=nd1;
    s_v[entry_index+24]=(dhdx3*dhdx1+dhdy3*dhdy1)*wt*det_jac;

    i_v[entry_index+25]=nd3;
    j_v[entry_index+25]=nd2;
    s_v[entry_index+25]=(dhdx3*dhdx2+dhdy3*dhdy2)*wt*det_jac;

    i_v[entry_index+26]=nd3;
    j_v[entry_index+26]=nd3;
    s_v[entry_index+26]=(dhdx3*dhdx3+dhdy3*dhdy3)*wt*det_jac;


  }//if(tid<nnodes...
}


// -------------------- main program ---------------------------

int main(void){


  //read input data from files
  int nnodes;
  int nel;

  ifstream params("params.laplace2d",ios::in);
  params >> nnodes;
  params >> nel;
  params.close();

  Real * gcoord = new Real[nnodes*2];
  int * nodes = new int[nel*3];

  ifstream gcoord_dat("gcoord.laplace2d",ios::in);
  for(int nd=0;nd<nnodes;nd++){
    gcoord_dat >> gcoord[nd]; //read x-value
    gcoord_dat >> gcoord[nnodes+nd];//read y-value
  }
  gcoord_dat.close();

  ifstream nodes_dat("nodes.laplace2d",ios::in);
  for(int el=0;el<nel;el++){
    nodes_dat >> nodes[el];//local node 1
    nodes_dat >> nodes[nel+el];//local node 2
    nodes_dat >> nodes[2*nel+el];//local node 3
  }
  nodes_dat.close();
  int num_entries = nel*27; //(NEL) x (3 Gauss points) x (3 i-nodes) x (3 j-nodes)
  //transfer data to the GPU
  Real * gcoord_d;
  int * nodes_d;
  cudaMalloc((void**)&gcoord_d,(nnodes*2)*sizeof(Real));
  cudaMalloc((void**)&nodes_d,(nel*3)*sizeof(int));

  cudaMemcpy(gcoord_d,gcoord,(nnodes*2)*sizeof(Real),cudaMemcpyHostToDevice);
  cudaMemcpy(nodes_d,nodes,(nel*3)*sizeof(int),cudaMemcpyHostToDevice);
  
  //construct raw vector entries for sparse matrices
 
  thrust::device_vector<int> j_v(num_entries);
  thrust::device_vector<int> i_v(num_entries);
  thrust::device_vector<Real> s_v(num_entries);

  int * i_p = thrust::raw_pointer_cast(&i_v[0]);
  int * j_p = thrust::raw_pointer_cast(&j_v[0]);
  Real * s_p = thrust::raw_pointer_cast(&s_v[0]);

  dim3 GRIDS((nel+BLOCK_SZ-1)/BLOCK_SZ,1,1);
  dim3 BLOCKS(BLOCK_SZ,1,1);

  //----- START TIMING ---------------------------------------------------

  clock_t begin = clock();


  compute_vectors<<<GRIDS,BLOCKS>>>(i_p,j_p,s_p,gcoord_d,nodes_d,
				    nnodes,nel);

  //sort the vectors and return the number of non-zeros

  int nnz = sort_and_combine<Real>(i_v,j_v,s_v,num_entries,nnodes,nnodes);

  //declare/allocate sparse matrix object
  cusp::csr_matrix<int,Real,cusp::device_memory> L_mat(nnodes,nnodes,nnz);

  //copy relevant vector data to the sparse matrix object
  copy_to_csr<Real>(L_mat,i_v,j_v,s_v,nnodes,nnodes,nnz);

  clock_t end = clock();
  //----------- STOP TIMING -------------------------------------------

  cout << "Time elapsed for matrix construction = "
       << ((double)end - (double)begin)/((double)CLOCKS_PER_SEC)
       << endl;

  //write the sparse matrix data to disk. (MATLAB code will check for 
  //correctness)

  cusp::io::write_matrix_market_file(L_mat,"L_cuda.mtx");

  //clean-up
  delete [] gcoord;
  delete [] nodes;
  cudaFree(gcoord_d);
  cudaFree(nodes_d);
  //remaining cleanup handled by CUSP/THRUST
  return 0;
}
