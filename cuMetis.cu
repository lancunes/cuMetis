
#include"include/cuMetis.h"

int test_time;

/*Time function params*/
double part_all = 0;
struct timeval begin_part_all;
struct timeval   end_part_all;

double part_coarsen = 0;
struct timeval begin_part_coarsen;
struct timeval   end_part_coarsen;

double part_init = 0;
struct timeval begin_part_init;
struct timeval   end_part_init;

double part_uncoarsen = 0;
struct timeval begin_part_uncoarsen;
struct timeval   end_part_uncoarsen;

//four calculation pattern
double part_match = 0;
struct timeval begin_part_match;
struct timeval   end_part_match;

double part_contract = 0;
struct timeval begin_part_contract;
struct timeval   end_part_contract;
  


double part_cmatch = 0;
struct timeval begin_part_cmatch;
struct timeval   end_part_cmatch;

double part_ccontract = 0;
struct timeval begin_part_ccontract;
struct timeval   end_part_ccontract;

double part_bfs = 0;
struct timeval begin_part_bfs;
struct timeval   end_part_bfs;

double part_2refine = 0;
struct timeval begin_part_2refine;
struct timeval   end_part_2refine;

double part_2map = 0;
struct timeval begin_part_2map;
struct timeval   end_part_2map;

double part_slipt = 0;
struct timeval begin_part_slipt;
struct timeval   end_part_slipt;



double part_krefine = 0;
struct timeval begin_part_krefine;
struct timeval   end_part_krefine;

double part_map = 0;
struct timeval begin_part_map;
struct timeval   end_part_map;

//match
double scuda_match = 0;
struct timeval begin_cuda_match;
struct timeval   end_cuda_match;

double scuda_cleanv = 0;
struct timeval begin_cuda_cleanv;
struct timeval   end_cuda_cleanv;

double sfindc1 = 0;
struct timeval begin_findc1;
struct timeval   end_findc1;

double sfindc2 = 0;
struct timeval begin_findc2;
struct timeval   end_findc2;

double sinclusive_scan = 0;
struct timeval begin_inclusive_scan;
struct timeval   end_inclusive_scan;

double sfindc2_5 = 0;
struct timeval begin_findc2_5;
struct timeval   end_findc2_5;

double sfindc3 = 0;
struct timeval begin_findc3;
struct timeval   end_findc3;

double sfindc4 = 0;
struct timeval begin_findc4;
struct timeval   end_findc4;


double sexclusive_scan = 0;
struct timeval begin_exclusive_scan;
struct timeval   end_exclusive_scan;

double sfind_cnvtxsedge_original = 0;
struct timeval begin_find_cnvtxsedge_original;
struct timeval   end_find_cnvtxsedge_original;

double sbb_segsort = 0;
struct timeval begin_bb_segsort;
struct timeval   end_bb_segsort;

double sSort_cnedges_part2 = 0;
struct timeval begin_Sort_cnedges_part2;
struct timeval   end_Sort_cnedges_part2;

double sinclusive_scan2 = 0;
struct timeval begin_inclusive_scan2;
struct timeval   end_inclusive_scan2;

double sinitcuda_match = 0;
struct timeval begin_initcuda_match;
struct timeval   end_initcuda_match;

double sinitcudajs = 0;
struct timeval begin_initcudajs;
struct timeval   end_initcudajs;

double sSort_cnedges_part1 = 0;
struct timeval begin_Sort_cnedges_part1;
struct timeval   end_Sort_cnedges_part1;

double sSort_cnedges_part2_5 = 0;
struct timeval begin_Sort_cnedges_part2_5;
struct timeval   end_Sort_cnedges_part2_5;

double sSort_cnedges_part3 = 0;
struct timeval begin_Sort_cnedges_part3;
struct timeval   end_Sort_cnedges_part3;

double sCoarsen = 0;
struct timeval begin_sCoarsen;
struct timeval   end_sCoarsen;

/*Define functions*/
#define cuMetis_max(m,n) ((m)>=(n)?(m):(n))
#define cuMetis_min(m,n) ((m)>=(n)?(n):(m))
#define cuMetis_swap(m,n,temp) do{(temp)=(m);(m)=(n);(n)=(temp);} while(0) 
#define cuMetis_tocsr(i,n,c) do{for(i=1;i<n;i++)c[i]+= c[i-1];for(i=n;i>0;i--)c[i]=c[i-1];c[0]=0;} while(0) 
#define cuMetis_add_sub(m,n,temp) do{(m)+=(temp);(n)-=(temp);} while(0)
#define cuMetis_listinsert(n,list,lptr,i) do{list[n]=i;lptr[i]=(n)++;} while(0) 
#define cuMetis_listdelete(n,list,lptr,i) do{list[lptr[i]]=list[--(n)];lptr[list[n]]=lptr[i];lptr[i]=-1;} while(0) 
#define M_GT_N(m,n) ((m)>(n))


/*Graph data structure*/
typedef struct cuMetis_graph_t {
  /*graph cpu params*/
  int nvtxs;                            //Graph vertex
  int nedges;	                          //Graph edge
  int *xadj;                            //Graph vertex csr array (xadj[nvtxs+1])
  int *adjncy;                          //Graph adjacency list (adjncy[nedges])
  int *adjwgt;   		                    //Graph edge weight array (adjwgt[nedges])
  int *vwgt;			                      //Graph vertex weight array(vwgr[nvtxs])
  int *tvwgt;                           //The sum of graph vertex weight 
  float *tvwgt_reverse;                 //The reciprocal of tvwgt
  int *label;                           //Graph vertex label(label[nvtxs])
  int *cmap;                            //The Label of graph vertex in cgraph(cmap[nvtxs]) 
  int mincut;                           //The min edfe-cut of graph partition
  int *where;                           //The label of graph vertex in which part(where[nvtxs]) 
  int *pwgts;                           //The partition vertex weight(pwgts[nparts])
  int nbnd;                             //Boundary vertex number
  int *bndlist;                         //Boundary vertex list
  int *bndptr;                          //Boundary vertex pointer
  int *id;                              //The sum of edge weight in same part
  int *ed;                              //The sum of edge weight in different part
  struct cuMetis_graph_t *coarser; //The coarser graph
  struct cuMetis_graph_t *finer;   //The finer graph
  /*graph gpu params*/
  int *cuda_nvtxs;
  int *cuda_xadj;
  int *cuda_adjncy;
  int *cuda_adjwgt;
  int *cuda_vwgt;               
  int *cuda_match;                      //CUDA graph vertex match array(match[nvtxs])
  int *cuda_cmap;
  int *cuda_maxvwgt;                    //CUDA graph constraint vertex weight 
  int *cuda_real_edge;                  //CUDA graph vertex pairs csr edge array(cuda_real_edge[cnvtxs+1])
  int *cuda_real_nvtxs;                 //CUDA graph params (i<match[cmap[i]])
  int *cuda_cnvtxs;                     //CUDA coarsen graph vertex
  int *cuda_s;                          //CUDA support array (cuda_s[nvtxs])
  int *cuda_scan_adjwgt_original;       //CUDA support scan array (cuda_scan_adjwgt_original[nedges])
  int *cuda_scan_nedges_original;       //CUDA support scan array (cuda_scan_nedges_original[nedges])
  int *cuda_js;                         //CUDA support array (cuda_js[cnvtxs])
  int *cuda_scan_cnedges_original;      //CUDA support scan array (cuda_scan_cnedges_original[nedges])
  int *cuda_maxwgt;                     //CUDA part weight array (cuda_maxwgt[npart])
  int *cuda_minwgt;                     //CUDA part weight array (cuda_minwgt[npart])
  int *cuda_where;
  int *cuda_pwgts;
  int *cuda_bnd;
  int *cuda_bndnum;
  int *cpu_bndnum;
  int *cuda_info;                       //CUDA support array(cuda_info[bnd_num*nparts])
  int *cuda_real_bnd_num;
  int *cuda_real_bnd;
  int *cuda_nparts;
  int *cuda_tvwgt;
  float *cuda_tpwgts;
} cuMetis_graph_t;

/*Refinement available generate array*/
int *cu_bn;                             
int *cu_bt;
int *cu_g;
int *cu_csr;
int *cu_que;

/*Memory allocation information*/
typedef struct cuMetis_mop_t {
  int type;
  ssize_t nbytes;
  void *ptr;
} cuMetis_mop_t;

/*Algorithm information*/
typedef struct cuMetis_mcore_t {
  void *core;	
  size_t coresize;     
  size_t corecpos;            
  size_t nmops;         
  size_t cmop;         
  cuMetis_mop_t *mops;      
  size_t num_callocs;   
  size_t num_hallocs;   
  size_t size_callocs;  
  size_t size_hallocs;  
  size_t cur_callocs;   
  size_t cur_hallocs;  
  size_t max_callocs;   
  size_t max_hallocs;   

} cuMetis_mcore_t;

/*Control information*/
typedef struct cuMetis_admin_t {
  int Coarsen_threshold;		
  int nIparts;                                                                                                                                       
  int iteration_num;                               
  int *maxvwgt;		                
  int nparts;                	
  float *ubfactors;            
  float *tpwgts;               
  float *part_balance;               
  float cfactor;               
  cuMetis_mcore_t *mcore;    
  size_t nbrpoolsize;      
  size_t nbrpoolcpos;                  

} cuMetis_admin_t;


/*Heap information*/
typedef struct cuMetis_rkv_t{
  float key;
  int val;
} cuMetis_rkv_t;


/*Queue information*/
typedef struct {
  ssize_t nnodes;
  ssize_t maxnodes;
  cuMetis_rkv_t   *heap;
  ssize_t *locator;
} cuMetis_queue_t;

/*Compute log2 algorithm*/
int cuMetis_compute_log2(int a)
{
  int i;
  for(i=1;a>1;i++,a=a>>1);
  return i-1;
}

/*Get int rand number*/
int cuMetis_int_rand() 
{
  if(sizeof(int)<=sizeof(int32_t)) 
    return (int)(uint32_t)rand();
  else  
    return (int)(uint64_t)rand(); 
}


/*Get int rand number between (0,max)*/
int cuMetis_int_randinrange(int max) 
{
  return (int)((cuMetis_int_rand())%max); 
}


/*Compute sum of int array*/
int cuMetis_int_sum(size_t n, int *a)
{
  size_t i;
  int sum=0;
  for(i=0;i<n;i++,a+=1){
    sum+=(*a);
  }
  return sum;
}

/*Copy int array a to b*/
int  *cuMetis_int_copy(size_t n, int *a, int *b)
{
  return (int *)memmove((void *)b, (void *)a, sizeof(int)*n);
}


/*Set int array value*/
int *cuMetis_int_set_value(size_t n, int val, int *a)
{
  size_t i;
  for(i=0;i<n;i++){
    a[i]=val;
  }
  return a;
}


/*Compute sum of float array*/
float cuMetis_float_sum(size_t n, float *a)
{
  size_t i;
  float sum=0;
  for(i=0;i<n;i++,a+=1){
    sum+=(*a);
  }
  return sum;
}


/*Rescale tpwgts array*/
float *cuMetis_tpwgts_rescale(size_t n, float wsum, float *a)
{
  size_t i;
  for(i=0;i<n;i++,a+=1){
    (*a)*=wsum;
  }
  return a;
}


/*Compute Partition result edge-cut*/
int cuMetis_computecut(cuMetis_graph_t *graph, int *where)
{
  int i,j,cut=0;
    for(i=0;i<graph->nvtxs;i++){
      for(j=graph->xadj[i];j<graph->xadj[i+1];j++)
        if(where[i]!=where[graph->adjncy[j]]){
          cut+=graph->adjwgt[j];
        }
    }
  return cut/2;
}


/*Set graph admin params*/
cuMetis_admin_t *cuMetis_set_graph_admin(int nparts, float *tpwgts, float *ubvec)
{
  int i;
  cuMetis_admin_t *cuMetis_admin;
  cuMetis_admin=(cuMetis_admin_t *)malloc(sizeof(cuMetis_admin_t));
  memset((void *)cuMetis_admin,0,sizeof(cuMetis_admin_t));

  cuMetis_admin->iteration_num=10;
  cuMetis_admin->Coarsen_threshold=200;
  cuMetis_admin->nparts=nparts; 

  cuMetis_admin->maxvwgt=(int*)malloc(sizeof(int));
  cuMetis_admin->maxvwgt[0]=0;  

  cuMetis_admin->tpwgts=(float*)malloc(sizeof(float)*nparts);
  for(i=0;i<nparts;i++){
    cuMetis_admin->tpwgts[i]=1.0/nparts;
  }

  cuMetis_admin->ubfactors=(float*)malloc(sizeof(float));
  cuMetis_admin->ubfactors[0] =1.03;

  cuMetis_admin->part_balance =(float*) malloc(sizeof(float)*nparts);
  return cuMetis_admin;  
}


/*Set graph params*/
void cuMetis_init_cpu_graph(cuMetis_graph_t *graph) 
{
  memset((void *)graph,0,sizeof(cuMetis_graph_t));
  graph->nvtxs     = -1;
  graph->nedges    = -1;
  graph->xadj      = NULL;
  graph->vwgt      = NULL;
  graph->adjncy    = NULL;
  graph->adjwgt    = NULL;
  graph->label     = NULL;
  graph->cmap      = NULL;
  graph->tvwgt     = NULL;
  graph->tvwgt_reverse  = NULL;
  graph->where     = NULL;
  graph->pwgts     = NULL;
  graph->mincut    = -1;
  graph->nbnd      = -1;
  graph->id        = NULL;
  graph->ed        = NULL;
  graph->bndptr    = NULL;
  graph->bndlist   = NULL;
  graph->coarser   = NULL;
  graph->finer     = NULL;
}


/*Malloc graph*/
cuMetis_graph_t *cuMetis_create_cpu_graph(void)
{
  cuMetis_graph_t *graph;
  graph=(cuMetis_graph_t *)malloc(sizeof(cuMetis_graph_t));
  cuMetis_init_cpu_graph(graph);
  return graph;
}


/*Set graph tvwgt value*/
void cuMetis_set_graph_tvwgt(cuMetis_graph_t *graph)
{
  if(graph->tvwgt==NULL){ 
    graph->tvwgt=(int*)malloc(sizeof(int));
  }

  if(graph->tvwgt_reverse==NULL){ 
    graph->tvwgt_reverse=(float*)malloc(sizeof(float));
  }

  graph->tvwgt[0]=cuMetis_int_sum(graph->nvtxs,graph->vwgt);
  graph->tvwgt_reverse[0]=1.0/(graph->tvwgt[0]>0?graph->tvwgt[0]:1);
}


/*Set graph vertex label*/
void cuMetis_set_graph_label(cuMetis_graph_t *graph)
{
  int i;

  if(graph->label==NULL){
    graph->label=(int*)malloc(sizeof(int)*(graph->nvtxs));
  }

  for(i=0;i<graph->nvtxs;i++){
    graph->label[i]=i;
  }

}


/*Set graph information*/
cuMetis_graph_t *cuMetis_set_graph(cuMetis_admin_t *cuMetis_admin, int nvtxs, \
int *xadj, int *adjncy, int *vwgt , int *adjwgt) 
{
  int i;
  cuMetis_graph_t *graph;
  
  graph = cuMetis_create_cpu_graph();
  graph->nvtxs=nvtxs;
  graph->nedges=xadj[nvtxs];
  graph->xadj=xadj;
  graph->adjncy=adjncy;
  
  if(vwgt){
    graph->vwgt=vwgt;
  }
  else{
    vwgt=graph->vwgt=(int*)malloc(sizeof(int)*nvtxs);
    for(i=0;i<nvtxs;i++){
      vwgt[i]=graph->vwgt[i]=1;
    }
  
  }
  
  graph->tvwgt=(int*)malloc(sizeof(int));
  graph->tvwgt_reverse=(float*)malloc(sizeof(float));
  graph->tvwgt[0]=cuMetis_int_sum(nvtxs, vwgt);
  graph->tvwgt_reverse[0]=1.0/(graph->tvwgt[0]>0?graph->tvwgt[0]:1);

  if(adjwgt){
    graph->adjwgt=adjwgt;
  }
  else{
    adjwgt=graph->adjwgt=(int*)malloc(sizeof(int)*(graph->nedges));
    for(i=0;i<graph->nedges;i++){
      adjwgt[i]=graph->adjwgt[i]=1;
    }
  }
  
  cuMetis_set_graph_tvwgt(graph);
  cuMetis_set_graph_label(graph);
  
  return graph;
}


/*Creates mcore*/
cuMetis_mcore_t *cuMetis_create_mcore(size_t coresize)
{
  cuMetis_mcore_t *mcore;
  mcore=(cuMetis_mcore_t *)malloc(sizeof(cuMetis_mcore_t));
  memset(mcore,0,sizeof(cuMetis_mcore_t));

  mcore->coresize=coresize;
  mcore->corecpos=0;
  mcore->core=(coresize==0?NULL:(size_t*)malloc(sizeof(size_t)*(mcore->coresize)));
  mcore->nmops=2048;
  mcore->cmop=0;
  mcore->mops=(cuMetis_mop_t *)malloc((mcore->nmops)*sizeof(cuMetis_mop_t));

  return mcore;
}


/*Allocate work space*/
void cuMetis_allocatespace(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  size_t coresize;
  coresize=3*(graph->nvtxs+1)*sizeof(int)+5*(cuMetis_admin->nparts+1)*sizeof(int)\
  +5*(cuMetis_admin->nparts+1)*sizeof(float);

  cuMetis_admin->mcore=cuMetis_create_mcore(coresize);
  cuMetis_admin->nbrpoolsize=0;
  cuMetis_admin->nbrpoolcpos=0;
}


/*Add memory allocation*/
void cuMetis_add_mcore(cuMetis_mcore_t *mcore, int type, size_t nbytes, void *ptr)
{
  if(mcore->cmop==mcore->nmops){
    mcore->nmops*=2;
    mcore->mops=(cuMetis_mop_t*)realloc(mcore->mops, mcore->nmops*sizeof(cuMetis_mop_t));
    if(mcore->mops==NULL){
      exit(0);
    }
  }

  mcore->mops[mcore->cmop].type=type;
  mcore->mops[mcore->cmop].nbytes=nbytes;
  mcore->mops[mcore->cmop].ptr=ptr;
  mcore->cmop++;

  switch(type){
    case 1:
      break;
    
    case 2:
      mcore->num_callocs++;
      mcore->size_callocs+=nbytes;
      mcore->cur_callocs+=nbytes;
      if(mcore->max_callocs<mcore->cur_callocs){
        mcore->max_callocs=mcore->cur_callocs;
      }
      break;
    
    case 3:
      mcore->num_hallocs++;
      mcore->size_hallocs+=nbytes;
      mcore->cur_hallocs+=nbytes;
      if(mcore->max_hallocs<mcore->cur_hallocs){
        mcore->max_hallocs=mcore->cur_hallocs;
      }
      break;
    
    default:
      exit(0);
  }
}


/*Malloc mcore*/
void *cuMetis_malloc_mcore(cuMetis_mcore_t *mcore, size_t nbytes)
{
  void *ptr;
  nbytes+=(nbytes%8==0?0:8-nbytes%8);

  if(mcore->corecpos+nbytes<mcore->coresize){
    ptr=((char *)mcore->core)+mcore->corecpos;
    mcore->corecpos+=nbytes;
    cuMetis_add_mcore(mcore,2,nbytes,ptr);
  }
  else{
    ptr=(size_t*)malloc(nbytes);
    cuMetis_add_mcore(mcore,3,nbytes,ptr);
  }

  return ptr;
}


/*Malloc mcore space*/
void *cuMetis_malloc_space(cuMetis_admin_t *cuMetis_admin, size_t nbytes)
{
  return cuMetis_malloc_mcore(cuMetis_admin->mcore,nbytes);
}


/*Malloc int mcore space*/
int *cuMetis_int_malloc_space(cuMetis_admin_t *cuMetis_admin, size_t n)
{
  return (int *)cuMetis_malloc_space(cuMetis_admin, n*sizeof(int));
}


/*Malloc float mcore space*/
float *cuMetis_float_malloc_space(cuMetis_admin_t *cuMetis_admin)
{
  return (float *)cuMetis_malloc_space(cuMetis_admin,2*sizeof(float));
}


/*Compute 2way balance params*/
void cuMetis_compute_2way_balance(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph, float *tpwgts)
{
  int i;
  for(i=0;i<2;i++){
      cuMetis_admin->part_balance[i]=graph->tvwgt_reverse[0]/tpwgts[i];
  }
}


/*Get random permute of p*/
void cuMetis_int_randarrayofp(int n, int *p, int m, int flag)
{
  int i,u,v;
  int temp;
  if(flag==1){
    for(i=0;i<n;i++)
      p[i] = (int)i;
  }

  if(n<10){
    for(i=0;i<n;i++){

      v=cuMetis_int_randinrange(n);
      u=cuMetis_int_randinrange(n);
     
      cuMetis_swap(p[v],p[u],temp);

    }
  }
  else{
    for(i=0;i<m;i++){

      v=cuMetis_int_randinrange(n-3);
      u=cuMetis_int_randinrange(n-3);
      
      cuMetis_swap(p[v+0],p[u+2],temp);
      cuMetis_swap(p[v+1],p[u+3],temp);
      cuMetis_swap(p[v+2],p[u+0],temp);
      cuMetis_swap(p[v+3],p[u+1],temp);

    }
  }
}


/*Get permutation array*/
void cuMetis_matching_sort(cuMetis_admin_t *cuMetis_admin, int n, \
int max, int *keys, int *tperm, int *perm)
{
  int i,ii;
  int *counts;
  counts=cuMetis_int_set_value(max+2,0,cuMetis_int_malloc_space(cuMetis_admin,max+2));
  
  for(i=0; i<n; i++){
    counts[keys[i]]++;
  }
  
  cuMetis_tocsr(i,max+1,counts);
  
  for(ii=0;ii<n;ii++){
    i=tperm[ii];
    perm[counts[keys[i]]++]=i;
  }
}


/*Malloc cpu coarsen graph params*/
cuMetis_graph_t *cuMetis_set_cpu_cgraph(cuMetis_graph_t *graph, int cnvtxs)
{
  cuMetis_graph_t *cgraph;
  cgraph=cuMetis_create_cpu_graph();
  
  cgraph->nvtxs=cnvtxs;
  cgraph->xadj=(int*)malloc(sizeof(int)*(cnvtxs+1));
  cgraph->adjncy=(int*)malloc(sizeof(int)*(graph->nedges));
  cgraph->adjwgt=(int*)malloc(sizeof(int)*(graph->nedges));
  cgraph->vwgt=(int*)malloc(sizeof(int)*cnvtxs);
  cgraph->tvwgt=(int*)malloc(sizeof(int));
  cgraph->tvwgt_reverse=(float*)malloc(sizeof(float)); 
  
  cgraph->finer=graph;
  graph->coarser=cgraph;
  
  return cgraph;
}



/*Malloc gpu coarsen graph params*/
cuMetis_graph_t *cuMetis_set_gpu_cgraph(cuMetis_graph_t *graph, int cnvtxs)
{
  cuMetis_graph_t *cgraph;
  cgraph=cuMetis_create_cpu_graph();
  
  cgraph->nvtxs=cnvtxs;
  cgraph->xadj=(int*)malloc(sizeof(int)*(cnvtxs+1));
  cgraph->tvwgt=(int*)malloc(sizeof(int));
  cgraph->tvwgt_reverse=(float*)malloc(sizeof(float)); 
  
  cgraph->finer=graph;
  graph->coarser=cgraph;
  
  return cgraph;
}



/*Create cpu coarsen graph by contract*/
void cuMetis_cpu_create_cgraph(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, int cnvtxs, int *match,int level)
{
  int j,k,m,istart,iend,nvtxs,nedges,cnedges,v,u;
  int *xadj,*vwgt,*adjncy,*adjwgt;
  int *cmap,*htable;
  int *cxadj,*cvwgt,*cadjncy,*cadjwgt;
  cuMetis_graph_t *cgraph;
  
  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  cmap=graph->cmap;                  
  
  cgraph=cuMetis_set_cpu_cgraph(graph,cnvtxs);            
  cxadj=cgraph->xadj;
  cvwgt=cgraph->vwgt;
  cadjncy=cgraph->adjncy;
  cadjwgt=cgraph->adjwgt;                               
  htable=cuMetis_int_set_value(cnvtxs,-1,cuMetis_int_malloc_space(cuMetis_admin,cnvtxs));      
  cxadj[0] = cnvtxs = cnedges = 0; 
  nedges=graph->nedges;
   
  for(v=0;v<nvtxs;v++){

    if((u=match[v])<v)         
      continue;   

    cvwgt[cnvtxs]=vwgt[v];                 
    nedges=0;                                                    
    istart=xadj[v];
    iend=xadj[v+1];    

    for(j=istart;j<iend;j++){

      k=cmap[adjncy[j]];     

      if((m=htable[k])==-1){
        cadjncy[nedges]=k;                           
        cadjwgt[nedges] = adjwgt[j];                      
        htable[k] = nedges++;  
      }
      else{
        cadjwgt[m] += adjwgt[j];                                 
      }
    }

    if(v!=u){ 
      cvwgt[cnvtxs]+=vwgt[u];                   
      istart=xadj[u];                                    
      iend=xadj[u+1];      

      for(j=istart;j<iend;j++){
        k=cmap[adjncy[j]];

        if((m=htable[k])==-1){
          cadjncy[nedges]=k;
          cadjwgt[nedges]=adjwgt[j];
          htable[k]=nedges++;
        }
        else{
          cadjwgt[m] += adjwgt[j];
        }
      }

      if((j=htable[cnvtxs])!=-1){
        cadjncy[j]=cadjncy[--nedges];
        cadjwgt[j]=cadjwgt[nedges];
        htable[cnvtxs] = -1;
      }
    }

    for(j=0;j<nedges;j++){
       htable[cadjncy[j]] = -1;  
    }

    cnedges+=nedges;
    cxadj[++cnvtxs]=cnedges;
    cadjncy+=nedges;                                                                 
    cadjwgt+=nedges;
  }

  cgraph->nedges=cnedges;
  cgraph->tvwgt[0]=cuMetis_int_sum(cgraph->nvtxs,cgraph->vwgt); 
  cgraph->tvwgt_reverse[0]=1.0/(cgraph->tvwgt[0]>0?cgraph->tvwgt[0]:1);    

}



/*Get cpu graph matching params by hem*/
int cuMetis_cpu_match(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph,int level)
{
  cudaDeviceSynchronize();
  gettimeofday(&begin_part_cmatch,NULL);

  int i,j,pi,k,nvtxs,cnvtxs,maxidx,maxwgt,aved;
  int *xadj,*vwgt,*adjncy,*adjwgt,*maxvwgt;
  int *match,*cmap,*d,*perm,*tperm;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  cmap=graph->cmap;
  maxvwgt=cuMetis_admin->maxvwgt;
  
  cnvtxs=0;
  match=cuMetis_int_set_value(nvtxs,-1, cuMetis_int_malloc_space(cuMetis_admin,nvtxs));
  perm=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  tperm=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  d=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);         
  cuMetis_int_randarrayofp(nvtxs,tperm,nvtxs/8,1);   
  aved=0.7*(xadj[nvtxs]/nvtxs);

  for(i=0;i<nvtxs;i++){ 
    d[i]=(xadj[i+1]-xadj[i]>aved?aved:xadj[i+1]-xadj[i]);
  }

  cuMetis_matching_sort(cuMetis_admin,nvtxs,aved,d,tperm,perm);         
  
  for(pi=0;pi<nvtxs;pi++) 
  {
    i=perm[pi];  

    if(match[i]==-1){  
      maxidx=i;                                                                               
      maxwgt=-1;           

      for(j=xadj[i];j<xadj[i+1];j++){
        k=adjncy[j];

        if(match[k]==-1&&maxwgt<adjwgt[j]&&vwgt[i]+vwgt[k]<=maxvwgt[0]){
          maxidx=k;
          maxwgt=adjwgt[j];
        }   

        if(maxidx==i&&3*vwgt[i]<maxvwgt[0]){ 
          maxidx=-1;
        }
      } 

      if(maxidx!=-1){
        cmap[i]=cmap[maxidx]=cnvtxs++;              
        match[i]=maxidx;                                        
        match[maxidx]=i; 
      }
    }
 
  }         

  for(cnvtxs=0,i=0;i<nvtxs;i++){
    if(match[i]==-1){
      match[i]=i;
      cmap[i]=cnvtxs++;                                                    
    }
    else{
      if(i<=match[i]){ 
        cmap[i]=cmap[match[i]]=cnvtxs++;
      }
    }
  }

  cudaDeviceSynchronize();
  gettimeofday(&end_part_cmatch,NULL);
  part_cmatch += (end_part_cmatch.tv_sec - begin_part_cmatch.tv_sec) * 1000 + (end_part_cmatch.tv_usec - begin_part_cmatch.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_part_ccontract,NULL);

  cuMetis_cpu_create_cgraph(cuMetis_admin, graph, cnvtxs, match,level);
  
  cudaDeviceSynchronize();
  gettimeofday(&end_part_ccontract,NULL);
  part_ccontract += (end_part_ccontract.tv_sec - begin_part_ccontract.tv_sec) * 1000 + (end_part_ccontract.tv_usec - begin_part_ccontract.tv_usec) / 1000.0;

  return cnvtxs;
}



/*Malloc and memcpy original graph from cpu to gpu*/
void cuMetis_malloc_original_coarseninfo(cuMetis_admin_t *cuMetis_admin,cuMetis_graph_t *graph)
{
    int nvtxs=graph->nvtxs;
    int nedges=graph->nedges;

    cudaMalloc((void**)&graph->cuda_nvtxs,sizeof(int));
    cudaMemcpy(graph->cuda_nvtxs,&graph->nvtxs,sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&graph->cuda_match,nvtxs*sizeof(int));

    cudaMalloc((void**)&graph->cuda_xadj,(nvtxs+1)*sizeof(int));
    cudaMemcpy(graph->cuda_xadj,graph->xadj,(nvtxs+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&graph->cuda_vwgt,(nvtxs+1)*sizeof(int));
    cudaMemcpy(graph->cuda_vwgt,graph->vwgt,nvtxs*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&graph->cuda_adjncy,nedges*sizeof(int));
    cudaMemcpy(graph->cuda_adjncy,graph->adjncy,nedges*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&graph->cuda_adjwgt,nedges*sizeof(int));
    cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,nedges*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&graph->cuda_cmap,nvtxs*sizeof(int));

    cudaMalloc((void**)&graph->cuda_maxvwgt,sizeof(int));

    cudaMalloc((void**)&graph->cuda_s,(nvtxs)*sizeof(int));

    cudaMalloc((void**)&graph->cuda_cnvtxs,sizeof(int));

    cudaMalloc((void**)&graph->cuda_scan_nedges_original,(graph->nedges)*sizeof(int));

    cudaMalloc((void**)&graph->cuda_scan_cnedges_original,(graph->nedges)*sizeof(int));

    cudaMalloc((void**)&graph->cuda_scan_adjwgt_original,(graph->nedges)*sizeof(int));

}



/*Malloc gpu coarsen graph params*/
  void cuMetis_malloc_coarseninfo(cuMetis_admin_t *cuMetis_admin,cuMetis_graph_t *graph)
{
    int nvtxs=graph->nvtxs;
    int nedges=graph->nedges;

    cudaMalloc((void**)&graph->cuda_nvtxs,sizeof(int));
    cudaMemcpy(graph->cuda_nvtxs,&graph->nvtxs,sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&graph->cuda_match,nvtxs*sizeof(int));

    cudaMalloc((void**)&graph->cuda_cmap,nvtxs*sizeof(int));

    cudaMalloc((void**)&graph->cuda_maxvwgt,sizeof(int)); 

    cudaMalloc((void**)&graph->cuda_s,(nvtxs)*sizeof(int));

    cudaMalloc((void**)&graph->cuda_cnvtxs,sizeof(int));

    cudaMalloc((void**)&graph->cuda_scan_nedges_original,nedges*sizeof(int));

    cudaMalloc((void**)&graph->cuda_scan_cnedges_original,nedges*sizeof(int));

    cudaMalloc((void**)&graph->cuda_scan_adjwgt_original,nedges*sizeof(int));

}


/*CUDA-initial cuda_js array*/
__global__ void initcudajs(int *cuda_js, int *cn)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<cn[0]){
  cuda_js[ii]=0;
  }

}



/*CUDA-set each vertex pair adjacency list and weight params*/
__global__ void find_cnvtxsedge_original(int *cuda_scan_nedges, int *cuda_scan_order, int *cuda_xadj,\
  int *cuda_match, int *cuda_adjncy, int *cuda_scan_nedges_original, int *cuda_cmap, int *cvwgt, int *vwgt, int *js,\
  int *cuda_scan_cnedges_original, int *cuda_scan_adjwgt_original, int *cuda_adjwgt)
{
  int pi,u,istart,iend,i,pp;
  pi=blockIdx.x;
  int tid=threadIdx.x;
  u=cuda_match[pi];  

  if(pi>u){
    pp=cuda_scan_nedges[cuda_cmap[pi]]+cuda_xadj[u+1]-cuda_xadj[u];
  }
  else{
    pp=cuda_scan_nedges[cuda_cmap[pi]];
  }

  int sum=(cuda_xadj[pi+1]-cuda_xadj[pi]);
  if(sum<32){

    if(threadIdx.x<sum){
      istart=cuda_xadj[pi]+threadIdx.x;
      iend=cuda_xadj[pi]+threadIdx.x+1;

      for(i=istart;i<iend;i++){
        int pt=pp+i-cuda_xadj[pi];
        cuda_scan_nedges_original[pt]=cuda_adjncy[i];
        cuda_scan_cnedges_original[pt]=cuda_cmap[cuda_adjncy[i]];
        cuda_scan_adjwgt_original[pt]=cuda_adjwgt[i];
      } 
    }

  }
  else{
    int tt=32;
    int b=sum/tt;
    int a=b+1;
    int x=sum-b*tt;

    if(threadIdx.x<x){
      istart=cuda_xadj[pi]+threadIdx.x*a;
      iend=cuda_xadj[pi]+(threadIdx.x+1)*a;
    }
    else{
      istart=cuda_xadj[pi]+x*a+(threadIdx.x-x)*b;
      iend=cuda_xadj[pi]+x*a+(threadIdx.x+1-x)*b;
    }
    for(i=istart;i<iend;i++){
      int pt=pp+i-cuda_xadj[pi];

      cuda_scan_nedges_original[pt]=cuda_adjncy[i];
      cuda_scan_cnedges_original[pt]=cuda_cmap[cuda_adjncy[i]];
      cuda_scan_adjwgt_original[pt]=cuda_adjwgt[i];
    } 
  } 
  if(tid==0){
    if(u!=pi){
      cvwgt[cuda_cmap[pi]]=vwgt[pi]+vwgt[u];
    }
    else{
      cvwgt[cuda_cmap[pi]]=vwgt[pi];
    }
  }
}


/*CUDA-Segmentation sorting part1-set scan array value 0 or 1*/
__global__ void Sort_cnedges2_part1(int *cuda_scan_cnedges_original,\
int *cuda_scan_nedges, int *cuda_scan_order, int *cuda_cmap, int *temp_scan)
{
  int pi,istart,iend,i;
  pi=blockIdx.x;
  int tid=threadIdx.x; 
  int pp,ppp;
  int pii;

  pii=cuda_scan_order[pi];
  pp=cuda_scan_nedges[cuda_cmap[pii]];
  ppp=cuda_scan_nedges[cuda_cmap[pii]+1];

  int sum=ppp-pp;

  if(sum<32){
    if(threadIdx.x<sum){
      istart=tid;
      iend=tid+1;

      for(i=istart;i<iend;i++){
        if(i==0){
          if(cuda_scan_cnedges_original[pp+i]==cuda_cmap[pii]){
            temp_scan[pp+i]=0;
          }
          else{
            temp_scan[pp+i]=1;
          }
        }
        else{
          if(cuda_scan_cnedges_original[pp+i]==cuda_cmap[pii]){
            temp_scan[pp+i]=0;
          }
          else{
            if(cuda_scan_cnedges_original[pp+i]==cuda_scan_cnedges_original[pp+i-1]){
              temp_scan[pp+i]=0;
            }
            else{
              temp_scan[pp+i]=1;
            }
          }
        }
      }
    }
  }
  else{
    int tt=32;
    int b=sum/tt;
    int a=b+1;
    int x=sum-b*tt;

    if(threadIdx.x<x){
      istart=threadIdx.x*a;
      iend=(threadIdx.x+1)*a;
    }
    else{
      istart=x*a+(threadIdx.x-x)*b;
      iend=x*a+(threadIdx.x+1-x)*b;
    }
    
    for(i=istart;i<iend;i++){
      if(i==0){
        if(cuda_scan_cnedges_original[pp+i]==cuda_cmap[pii]){
          temp_scan[pp+i]=0;
        }
        else{
          temp_scan[pp+i]=1;
        }
      }
      else{
        if(cuda_scan_cnedges_original[pp+i]==cuda_cmap[pii]){
          temp_scan[pp+i]=0;
        }
        else{
          if(cuda_scan_cnedges_original[pp+i]==cuda_scan_cnedges_original[pp+i-1]){
            temp_scan[pp+i]=0;
          }
          else{
            temp_scan[pp+i]=1;
          }
        }
      }
    }
  }
}


/*CUDA-Segmentation sorting part2-set cxadj*/
__global__ void Sort_cnedges2_part2(int *cuda_scan_nedges, int *cuda_scan_order, int *cuda_cmap, \
int *temp_scan, int *temp_xadj, int *cn)
{
  int pi;
  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<cn[0]){ 
    int ppp;
    int pii;  
    pii=cuda_scan_order[pi];
    ppp=cuda_scan_nedges[cuda_cmap[pii]+1];
      
  if(pi==0){
    temp_xadj[0]=0;
  }

  temp_xadj[pi+1]=temp_scan[ppp-1];

  }
} 


/*CUDA-Segmentation sorting part2.5-init cadjwgt and cadjncy*/
__global__ void Sort_cnedges2_part2_5(int *cadjwgt, int *cadjncy, int *c)
{
  int pi;
  pi=blockIdx.x*blockDim.x+threadIdx.x;  

  if(pi<c[0]){ 
    cadjwgt[pi]=0;
    cadjncy[pi]=-1;
  }
}


/*CUDA-Segmentation sorting part3-deduplication and accumulation*/
__global__ void Sort_cnedges2_part3(int *cuda_scan_cnedges_original,int *cuda_scan_nedges, \
int *cuda_scan_order, int *cuda_cmap,  int *cuda_scan_adjwgt_original,int *temp_scan, int *cadjncy, int *cadjwgt)
{
  int pi,istart,iend,i;
  pi=blockIdx.x;
  int tid=threadIdx.x;
  int pp,ppp;
  int pii;

  pii=cuda_scan_order[pi];
  pp=cuda_scan_nedges[cuda_cmap[pii]];
  ppp=cuda_scan_nedges[cuda_cmap[pii]+1];

  int sum=ppp-pp;

  if(sum<32){
    if(threadIdx.x<sum){
      istart=tid;
      iend=tid+1;

      for(i=istart;i<iend;i++){
        if(i==0){
          if(cuda_scan_cnedges_original[pp+i]!=cuda_cmap[pii]){
            cadjncy[temp_scan[pp+i]-1]=cuda_scan_cnedges_original[pp+i];
            atomicAdd(&cadjwgt[temp_scan[pp+i]-1],cuda_scan_adjwgt_original[pp+i]);
          }
        }
        else{
          if(cuda_scan_cnedges_original[pp+i]!=cuda_cmap[pii]){
            if(cuda_scan_cnedges_original[pp+i]!=cuda_scan_cnedges_original[pp+i-1]){
              cadjncy[temp_scan[pp+i]-1]=cuda_scan_cnedges_original[pp+i];
              atomicAdd(&cadjwgt[temp_scan[pp+i]-1],cuda_scan_adjwgt_original[pp+i]);
            }
            else{
              atomicAdd(&cadjwgt[temp_scan[pp+i]-1],cuda_scan_adjwgt_original[pp+i]);
            }
          }
        }   
      }
    }
  }
  else {
    int tt=32;
    int b=sum/tt;
    int a=b+1;
    int x=sum-b*tt;

    if(threadIdx.x<x){
      istart=threadIdx.x*a;
      iend=(threadIdx.x+1)*a;
    }
    else{
      istart=x*a+(threadIdx.x-x)*b;
      iend=x*a+(threadIdx.x+1-x)*b;
    }
    for(i=istart;i<iend;i++){
      if(i==0){
        if(cuda_scan_cnedges_original[pp+i]!=cuda_cmap[pii]){
          cadjncy[temp_scan[pp+i]-1]=cuda_scan_cnedges_original[pp+i];
          atomicAdd(&cadjwgt[temp_scan[pp+i]-1],cuda_scan_adjwgt_original[pp+i]);
        }
      }
      else{
        if(cuda_scan_cnedges_original[pp+i]!=cuda_cmap[pii]){
          if(cuda_scan_cnedges_original[pp+i]!=cuda_scan_cnedges_original[pp+i-1]){
            cadjncy[temp_scan[pp+i]-1]=cuda_scan_cnedges_original[pp+i];
            atomicAdd(&cadjwgt[temp_scan[pp+i]-1],cuda_scan_adjwgt_original[pp+i]);
          }
          else{
            atomicAdd(&cadjwgt[temp_scan[pp+i]-1],cuda_scan_adjwgt_original[pp+i]);
          }
        }
      }     
    }
  }
}


/*Free cuda coarsen graph params*/
void cuMetis_free_coarsen(cuMetis_graph_t *graph)
{
  cudaFree(graph->cuda_maxvwgt);
  cudaFree(graph->cuda_match);
  cudaFree(graph->cuda_real_edge);
  cudaFree(graph->cuda_real_nvtxs);
  cudaFree(graph->cuda_s);
  cudaFree(graph->cuda_scan_adjwgt_original);
  cudaFree(graph->cuda_scan_nedges_original);
  cudaFree(graph->cuda_js);
  cudaFree(graph->cuda_scan_cnedges_original);
}



/*Create gpu coarsen graph by contract*/
void cuMetis_gpu_create_cgraph(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, int cnvtxs, int level,int *scan_edge)
{
  int nvtxs=graph->nvtxs;
  int nedges=graph->nedges;

  cuMetis_graph_t *cgraph;
  cgraph = cuMetis_set_gpu_cgraph(graph, cnvtxs); 

  int length=cnvtxs+1; 

  cudaDeviceSynchronize();
  gettimeofday(&begin_exclusive_scan,NULL);
  thrust::exclusive_scan(scan_edge,scan_edge+length,scan_edge);
  cudaDeviceSynchronize();//计算临界边 索引
  gettimeofday(&end_exclusive_scan,NULL);
  sexclusive_scan += (end_exclusive_scan.tv_sec - begin_exclusive_scan.tv_sec) * 1000 + (end_exclusive_scan.tv_usec - begin_exclusive_scan.tv_usec) / 1000.0;

  cudaMemcpy( graph->cuda_real_edge, scan_edge, (cnvtxs+1)* sizeof(int), cudaMemcpyHostToDevice);
  
  cudaMalloc((void**)&cgraph->cuda_vwgt, cnvtxs*sizeof(int));  
  cudaMalloc((void**)&graph->cuda_js, cnvtxs*sizeof(int));

  cudaDeviceSynchronize();
  gettimeofday(&begin_initcudajs,NULL);
  initcudajs<<<cnvtxs/32+1,32>>>(graph->cuda_js,graph->cuda_cnvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_initcudajs,NULL);
  sinitcudajs += (end_initcudajs.tv_sec - begin_initcudajs.tv_sec) * 1000 + (end_initcudajs.tv_usec - begin_initcudajs.tv_usec) / 1000.0;
  
  cudaDeviceSynchronize();
  gettimeofday(&begin_find_cnvtxsedge_original,NULL);
  find_cnvtxsedge_original<<<nvtxs,32>>>(graph->cuda_real_edge,graph->cuda_real_nvtxs,graph->cuda_xadj,graph->cuda_match,\
    graph->cuda_adjncy,graph->cuda_scan_nedges_original,graph->cuda_cmap,cgraph->cuda_vwgt,graph->cuda_vwgt,graph->cuda_js,\
    graph->cuda_scan_cnedges_original,graph->cuda_scan_adjwgt_original,graph->cuda_adjwgt);
  cudaDeviceSynchronize();
  gettimeofday(&end_find_cnvtxsedge_original,NULL);
  sfind_cnvtxsedge_original += (end_find_cnvtxsedge_original.tv_sec - begin_find_cnvtxsedge_original.tv_sec) * 1000 + (end_find_cnvtxsedge_original.tv_usec - begin_find_cnvtxsedge_original.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_bb_segsort,NULL);
  bb_segsort(graph->cuda_scan_cnedges_original, graph->cuda_scan_adjwgt_original, graph->nedges, graph->cuda_real_edge, cnvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_bb_segsort,NULL);
  sbb_segsort += (end_bb_segsort.tv_sec - begin_bb_segsort.tv_sec) * 1000 + (end_bb_segsort.tv_usec - begin_bb_segsort.tv_usec) / 1000.0;

  int *temp_scan;
  cudaMalloc((void**)&temp_scan, nedges*sizeof(int));

  cudaDeviceSynchronize();
  gettimeofday(&begin_Sort_cnedges_part1,NULL);
  Sort_cnedges2_part1<<<cnvtxs,32>>>(graph->cuda_scan_cnedges_original,graph->cuda_real_edge,graph->cuda_real_nvtxs,\
    graph->cuda_cmap,temp_scan);
  cudaDeviceSynchronize();
  gettimeofday(&end_Sort_cnedges_part1,NULL);
  sSort_cnedges_part1 += (end_Sort_cnedges_part1.tv_sec - begin_Sort_cnedges_part1.tv_sec) * 1000 + (end_Sort_cnedges_part1.tv_usec - begin_Sort_cnedges_part1.tv_usec) / 1000.0;
  
  cudaDeviceSynchronize();
  gettimeofday(&begin_inclusive_scan,NULL);
  thrust::device_ptr<int> ccscan = thrust::device_pointer_cast<int>(temp_scan);
  thrust::inclusive_scan(ccscan,ccscan+nedges,ccscan);
  cudaDeviceSynchronize();
  gettimeofday(&end_inclusive_scan,NULL);
  sinclusive_scan += (end_inclusive_scan.tv_sec - begin_inclusive_scan.tv_sec) * 1000 + (end_inclusive_scan.tv_usec - begin_inclusive_scan.tv_usec) / 1000.0;

  cudaMalloc((void**)&cgraph->cuda_xadj, (cnvtxs+1)*sizeof(int));

  cudaDeviceSynchronize();
  gettimeofday(&begin_Sort_cnedges_part2,NULL);
  Sort_cnedges2_part2<<<cnvtxs/32+1,32>>>(graph->cuda_real_edge,graph->cuda_real_nvtxs,\
    graph->cuda_cmap,temp_scan,cgraph->cuda_xadj,graph->cuda_cnvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_Sort_cnedges_part2,NULL);
  sSort_cnedges_part2 += (end_Sort_cnedges_part2.tv_sec - begin_Sort_cnedges_part2.tv_sec) * 1000 + (end_Sort_cnedges_part2.tv_usec - begin_Sort_cnedges_part2.tv_usec) / 1000.0;
  
  int *cxadj=(int *)malloc(sizeof(int)*(cnvtxs+1));
  cudaMemcpy(cxadj, cgraph->cuda_xadj, (cnvtxs+1)* sizeof(int), cudaMemcpyDeviceToHost); 

  cgraph->nvtxs=cnvtxs;
  cgraph->nedges=cxadj[cnvtxs];

  cudaMalloc((void**)&cgraph->cuda_adjncy,   cgraph->nedges*sizeof(int));
  cudaMalloc((void**)&cgraph->cuda_adjwgt,   cgraph->nedges*sizeof(int));

  int *ccc;
  cudaMalloc((void**)&ccc,sizeof(int));
  cudaMemcpy(ccc, &cgraph->nedges, sizeof(int), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  gettimeofday(&begin_Sort_cnedges_part2_5,NULL);
  Sort_cnedges2_part2_5<<<(cgraph->nedges/32)+1,32>>>(cgraph->cuda_adjwgt,cgraph->cuda_adjncy,ccc);
  cudaDeviceSynchronize();
  gettimeofday(&end_Sort_cnedges_part2_5,NULL);
  sSort_cnedges_part2_5 += (end_Sort_cnedges_part2_5.tv_sec - begin_Sort_cnedges_part2_5.tv_sec) * 1000 + (end_Sort_cnedges_part2_5.tv_usec - begin_Sort_cnedges_part2_5.tv_usec) / 1000.0;

  //累加 权重 点3：5+1=6
  cudaDeviceSynchronize();
  gettimeofday(&begin_Sort_cnedges_part3,NULL);
  Sort_cnedges2_part3<<<cnvtxs,32>>>(graph->cuda_scan_cnedges_original,graph->cuda_real_edge,graph->cuda_real_nvtxs,\
    graph->cuda_cmap,graph->cuda_scan_adjwgt_original,temp_scan,\
    cgraph->cuda_adjncy,cgraph->cuda_adjwgt);
  cudaDeviceSynchronize();
  gettimeofday(&end_Sort_cnedges_part3,NULL);
  sSort_cnedges_part3 += (end_Sort_cnedges_part3.tv_sec - begin_Sort_cnedges_part3.tv_sec) * 1000 + (end_Sort_cnedges_part3.tv_usec - begin_Sort_cnedges_part3.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  cudaFree(ccc);
  cudaFree(temp_scan);

  cgraph->tvwgt[0]=graph->tvwgt[0];   

  if(level!=0){
    cuMetis_free_coarsen(graph);
  }

}



/*CUDA-init match array*/
__global__ void initcuda_match(int *cuda_match,int *a)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
  cuda_match[ii]=-1;
  }
}



/*CUDA-hem matching*/
__global__ void cuda_hem(int *cuda_nvtxs, int *match, int *xadj, int *vwgt,\
int *adjwgt, int *adjncy, int *maxvwgt)
{
  int pi;
  int ii;
  int i,j,k,maxidx,maxwgt;
  ii=blockIdx.x*blockDim.x+threadIdx.x;
  int b_start,b_end;
  int tt=1024;
  int nvtxs=cuda_nvtxs[0];

  if(nvtxs%tt==0){
    b_start=ii*(nvtxs/tt);
    b_end=(ii+1)*(nvtxs/tt);
  }
  else{
    int b=nvtxs/tt;
    int a=b+1;
    int x=nvtxs-b*tt;

    if(ii<x){
      b_start=ii*a;
      b_end=(ii+1)*a;
    }
    else{
      b_start=x*a+(ii-x)*b;
      b_end=x*a+(ii+1-x)*b;
    }
  }

  for(pi=b_start;pi<b_end;pi++){
    i=pi;

    if(match[i]==-1){  
      maxidx=i;                                                                               
      maxwgt=-1;       

      for(j=xadj[i];j<xadj[i+1];j++){
        k=adjncy[j];

        if(match[k]==-1&&maxwgt<adjwgt[j]&&vwgt[i]+vwgt[k]<=maxvwgt[0]){
          maxidx=k;
          maxwgt=adjwgt[j];
        }  
        if(maxidx==i&&3*vwgt[i]<maxvwgt[0]){ 
          maxidx = -1;
        }
      }
      if(maxidx!=-1){    
        match[i] = maxidx;  
        atomicExch(&match[maxidx],i);                                 
      }
    }
  }
}


//edge
__global__ void cuda_shem1(int *nvtxs, int *xadj, int *adjwgt, int *match, int *adjncy, int *vwgt, int *maxvwgt)
{
  int ii = blockIdx.x;
  int j, k, maxidx, maxwgt;

  // do
  // {
  //    __threadfence_block();
  //    __threadfence();
  // }while()
  if(threadIdx.x == 0)
  {
    maxidx = ii;                                                                               
    maxwgt = -1; 

    int cnt = xadj[ii + 1] - xadj[ii];

    do
    {
      if(cnt == 0) break;
      // printf("maxvwgt=%d\n",maxvwgt[0]);
      for(j = xadj[ii];j < xadj[ii + 1];++j)
      {
        k = adjncy[j];
        if (match[k] < 0 && maxwgt < adjwgt[j] && vwgt[ii]+vwgt[k] <= maxvwgt[0]) 
        {
          maxidx = k;
          maxwgt = adjwgt[j];
        }
        if (maxidx == ii && 3 * vwgt[ii] < maxvwgt[0]) 
        { 
          maxidx = -1;
        }
      }
      /*if (maxidx != -1) 
      {
        if(match[ii] == -1) match[ii] = maxidx;

        __syncthreads();

        if(match[maxidx] == ii) break;
        else cnt--;
      }
      else break;*/
      // printf("ii=%d maxidx=%d\n",ii,maxidx);
      if(maxidx != -1) 
      {
        __threadfence();
        // if(atomicCAS(&match[ii],-1,maxidx))
        if(match[ii] == -1)
        {
          // atomicExch(&match[ii],maxidx);
          __threadfence();
          // if(atomicCAS(&match[maxidx],-1,ii)) 
          if(match[maxidx] == -1)
          {
            match[ii] = maxidx;
            match[maxidx] = ii;
            // atomicExch(&match[ii],maxidx);
            // atomicExch(&match[maxidx],ii);
            // printf("--ii=%d match[%d]=%d\n",ii,ii,match[ii]);
            break;
          }
          else if(match[maxidx] != ii) 
          {
            // atomicExch(&match[ii],-1); 
            cnt--;
          }
          else break;
        } 
        else break;
      }
      else break;
    }while(match[ii] == -1);

    // printf("match[%d]=%d\n",ii,match[ii]);
  }
}



__global__ void cuda_shem2(int *nvtxs, int *xadj, int *adjwgt, int *match, int *adjncy, int *vwgt, int *maxvwgt)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k, maxidx, maxwgt;

  if(ii < nvtxs[0])
  {
    if(3 * vwgt[ii] < maxvwgt[0])
    {
      for(j = xadj[ii + 1] - 1;j >= xadj[ii];--j)
      {
        k = adjncy[j];

        if(match[ii] != -1) break;
        else if(match[k] == ii)
        {
          match[ii] = k;
          break;
        }
        else if(match[k] != -1)
          continue;
        else if(vwgt[ii]+vwgt[k] > maxvwgt[0])
          continue;
        else
        {
          match[ii] = k;
          match[k] = ii;
          break;
        }
      }
    }
  }
}



/*CUDA-set conflict array*/
__global__ void cuda_cleanv(int *match, int *s, int *a)
{
  int pi,u;
  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<a[0]){
    s[pi]=1;

    if(match[pi]!=-1){
      u=match[pi];

      if(match[u]!=pi){
        s[pi]=0;
      }
    }
  } 
} 



/*CUDA-find cgraph vertex part1-remark the match array by s*/
__global__ void findc1(int *cuda_match, int *cuda_cmap, int *cuda_nvtxs, int *s)
{
  int pi;
  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<cuda_nvtxs[0]){
    if(s[pi]==0||cuda_match[pi] == -1)
    cuda_match[pi]=pi;
  }
}


/*CUDA-find cgraph vertex part2-make sure the pair small label vertex*/
__global__ void findc2(int *cuda_match, int *cuda_cmap, int *cuda_nvtxs, int *s)
{
  int pi;
  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<cuda_nvtxs[0]){
    if(pi<=cuda_match[pi]){
      cuda_cmap[pi]=1;
    }
    else{
      cuda_cmap[pi]=0;
    }
  }
}


/*CUDA-find cgraph vertex part2.5-init array*/
__global__ void findc2_5(int *cuda_temp, int *cuda_cmap, int *nvtxs)
{
  int i=nvtxs[0]-1;
  cuda_temp[0]=cuda_cmap[i];
}



/*CUDA-find cgraph vertex part3-array reduce 1*/
__global__ void findc3(int *cuda_match, int *cuda_cmap, int *cuda_nvtxs, int *s)
{
  int pi;
  pi=blockIdx.x*blockDim.x+threadIdx.x;
  if(pi<cuda_nvtxs[0]){
    cuda_cmap[pi]--;
  }
}


/*CUDA-find cgraph vertex part4-make sure vertex pair real rdge*/
__global__ void findc4(int *cuda_match, int *real, int *cmap, int *cnedges, int *xadj, int *nvtxs)
{
  int pi;
  int u;

  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<nvtxs[0]){
    if(pi>cuda_match[pi]){
      cmap[pi]=cmap[cuda_match[pi]];
    }
    else{
      u=cuda_match[pi];
      real[cmap[pi]]=pi;

      if(u!=pi){
        cnedges[cmap[pi]]=(xadj[pi+1]-xadj[pi])+(xadj[u+1]-xadj[u]);
      }
      else{
        cnedges[cmap[pi]]=(xadj[pi+1]-xadj[pi]);
      }
    }
  }
}



/*Get gpu graph matching params by hem*/
int cuMetis_gpu_match(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph, int level)
{
  cudaDeviceSynchronize();
  gettimeofday(&begin_part_match,NULL);

  int nvtxs  = graph->nvtxs;
  int nedges = graph->nedges;

  cudaDeviceSynchronize();
  gettimeofday(&begin_initcuda_match,NULL);
  initcuda_match<<<nvtxs/32+1,32>>>(graph->cuda_match,graph->cuda_nvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_initcuda_match,NULL);
  sinitcuda_match += (end_initcuda_match.tv_sec - begin_initcuda_match.tv_sec) * 1000 + (end_initcuda_match.tv_usec - begin_initcuda_match.tv_usec) / 1000.0;

  cudaMemcpy(  graph->cuda_maxvwgt, cuMetis_admin->maxvwgt, sizeof(int), cudaMemcpyHostToDevice); 

  cudaDeviceSynchronize();
  gettimeofday(&begin_cuda_match,NULL);
  // cuda_hem<<<1024,1>>>(graph->cuda_nvtxs,graph->cuda_match,graph->cuda_xadj,\
  //   graph-> cuda_vwgt,graph->cuda_adjwgt,graph->cuda_adjncy,graph->cuda_maxvwgt);
  // cuda_shem1<<<nvtxs,1>>>(graph->cuda_nvtxs,graph->cuda_xadj,graph->cuda_adjwgt,\
  //   graph->cuda_match,graph->cuda_adjncy,graph-> cuda_vwgt,graph->cuda_maxvwgt);
  
  bb_segsort(graph->cuda_adjwgt,graph->cuda_adjncy,nedges,graph->cuda_xadj,nvtxs);

  cuda_shem2<<<nvtxs/32+1,32>>>(graph->cuda_nvtxs,graph->cuda_xadj,graph->cuda_adjwgt,\
    graph->cuda_match,graph->cuda_adjncy,graph-> cuda_vwgt,graph->cuda_maxvwgt);

  cudaDeviceSynchronize();
  gettimeofday(&end_cuda_match,NULL);
  scuda_match += (end_cuda_match.tv_sec - begin_cuda_match.tv_sec) * 1000 + (end_cuda_match.tv_usec - begin_cuda_match.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_cuda_cleanv,NULL);
  cuda_cleanv<<<nvtxs/32+1,32>>>(graph->cuda_match,graph->cuda_s,graph->cuda_nvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_cuda_cleanv,NULL);
  scuda_cleanv += (end_cuda_cleanv.tv_sec - begin_cuda_cleanv.tv_sec) * 1000 + (end_cuda_cleanv.tv_usec - begin_cuda_cleanv.tv_usec) / 1000.0;

  int cnvtxs=0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_findc1,NULL);
  findc1<<<nvtxs/32+1,32>>>(graph->cuda_match,graph->cuda_cmap,graph->cuda_nvtxs,graph->cuda_s);
  cudaDeviceSynchronize();
  gettimeofday(&end_findc1,NULL);
  sfindc1 += (end_findc1.tv_sec - begin_findc1.tv_sec) * 1000 + (end_findc1.tv_usec - begin_findc1.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_findc2,NULL);
  findc2<<<nvtxs/32+1,32>>>(graph->cuda_match,graph->cuda_cmap,graph->cuda_nvtxs,graph->cuda_s);
  cudaDeviceSynchronize();
  gettimeofday(&end_findc2,NULL);
  sfindc2 += (end_findc2.tv_sec - begin_findc2.tv_sec) * 1000 + (end_findc2.tv_usec - begin_findc2.tv_usec) / 1000.0;

  thrust::device_ptr<int> cscan = thrust::device_pointer_cast<int>(graph->cuda_cmap);
  cudaDeviceSynchronize();
  gettimeofday(&begin_inclusive_scan2,NULL);
  thrust::inclusive_scan(cscan,cscan+nvtxs,cscan);
  cudaDeviceSynchronize();
  gettimeofday(&end_inclusive_scan2,NULL);
  sinclusive_scan2 += (end_inclusive_scan2.tv_sec - begin_inclusive_scan2.tv_sec) * 1000 + (end_inclusive_scan2.tv_usec - begin_inclusive_scan2.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_findc2_5,NULL);
  findc2_5<<<1,1>>>(graph->cuda_cnvtxs,graph->cuda_cmap,graph->cuda_nvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_findc2_5,NULL);
  sfindc2_5 += (end_findc2_5.tv_sec - begin_findc2_5.tv_sec) * 1000 + (end_findc2_5.tv_usec - begin_findc2_5.tv_usec) / 1000.0;
  
  cudaMemcpy(  &cnvtxs,graph->cuda_cnvtxs,  sizeof(int), cudaMemcpyDeviceToHost);
  
  int *cpu_scan_edge=(int *)malloc(sizeof(int)*(cnvtxs+1));
  cudaMalloc((void**)&graph->cuda_real_nvtxs,  cnvtxs*sizeof(int));
  cudaMalloc((void**)&graph->cuda_real_edge,  (cnvtxs+1)*sizeof(int));

  cudaDeviceSynchronize();
  gettimeofday(&begin_findc3,NULL);
  findc3<<<nvtxs/32+1,32>>>(graph->cuda_match,graph->cuda_cmap,graph->cuda_nvtxs,graph->cuda_s);
  cudaDeviceSynchronize();
  gettimeofday(&end_findc3,NULL);
  sfindc3 += (end_findc3.tv_sec - begin_findc3.tv_sec) * 1000 + (end_findc3.tv_usec - begin_findc3.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_findc4,NULL);
  findc4<<<nvtxs/32+1,32>>>(graph->cuda_match,graph->cuda_real_nvtxs,graph->cuda_cmap,graph->cuda_real_edge,graph->cuda_xadj,graph->cuda_nvtxs);
  cudaDeviceSynchronize();//预估粗点临界边
  gettimeofday(&end_findc4,NULL);
  sfindc4 += (end_findc4.tv_sec - begin_findc4.tv_sec) * 1000 + (end_findc4.tv_usec - begin_findc4.tv_usec) / 1000.0;
  cudaMemcpy(  cpu_scan_edge, graph->cuda_real_edge,  (cnvtxs+1)*sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();
  gettimeofday(&end_part_match,NULL);
  part_match += (end_part_match.tv_sec - begin_part_match.tv_sec) * 1000 + (end_part_match.tv_usec - begin_part_match.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_part_contract,NULL);

  cuMetis_gpu_create_cgraph(cuMetis_admin, graph, cnvtxs, level,cpu_scan_edge);  
  
  cudaDeviceSynchronize();
  gettimeofday(&end_part_contract,NULL);
  part_contract += (end_part_contract.tv_sec - begin_part_contract.tv_sec) * 1000 + (end_part_contract.tv_usec - begin_part_contract.tv_usec) / 1000.0;
  
  return cnvtxs;

}


void cuMetis_memcpy_coarsentoinit(cuMetis_graph_t *graph)
{
  int nvtxs=graph->nvtxs;
  int nedges=graph->nedges;
  graph->vwgt=(int *)malloc(sizeof(int)*nvtxs); 
  graph->adjncy=(int *)malloc(sizeof(int)*nedges);
  graph->adjwgt=(int *)malloc(sizeof(int)*nedges);
  cudaMemcpy(  graph->xadj, graph->cuda_xadj , (nvtxs+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(  graph->vwgt, graph->cuda_vwgt , (nvtxs)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(  graph->adjncy, graph->cuda_adjncy , (nedges)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(  graph->adjwgt, graph->cuda_adjwgt , (nedges)*sizeof(int), cudaMemcpyDeviceToHost);
}


/*Gpu multilevel coarsen*/
cuMetis_graph_t *cuMetis_coarsen(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  int level=0;

  cuMetis_admin->maxvwgt[0]=1.5*graph->tvwgt[0]/cuMetis_admin->Coarsen_threshold; 
  
  do{  
    if(level!=0){
      cuMetis_malloc_coarseninfo(cuMetis_admin,graph);
    }
    cudaDeviceSynchronize();
    gettimeofday(&begin_sCoarsen,NULL);
    cuMetis_gpu_match(cuMetis_admin,graph,level);   
    gettimeofday(&end_sCoarsen,NULL);
    sCoarsen += (end_sCoarsen.tv_sec - begin_sCoarsen.tv_sec) * 1000 + (end_sCoarsen.tv_usec - begin_sCoarsen.tv_usec) / 1000.0;

    graph = graph->coarser;   
    level++;       

  }while(graph->nvtxs>cuMetis_admin->Coarsen_threshold&&\
      graph->nvtxs<0.75*graph->finer->nvtxs&&\
      graph->nedges>graph->nvtxs/2); 

  cuMetis_memcpy_coarsentoinit(graph);  

  return graph;
}


/*Cpu multilevel coarsen*/
cuMetis_graph_t *cuMetis_cpu_coarsen(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  int level=1;

  cuMetis_admin->maxvwgt[0] = 1.5*graph->tvwgt[0]/cuMetis_admin->Coarsen_threshold;
 
  do{
    if(graph->cmap==NULL){
      graph->cmap=(int*)malloc(sizeof(int)*(graph->nvtxs));
    }

    cuMetis_cpu_match (cuMetis_admin,graph,level);

    graph = graph->coarser;
    level++;

  }while(graph->nvtxs > cuMetis_admin->Coarsen_threshold && 
      graph->nvtxs < 0.75*graph->finer->nvtxs && 
      graph->nedges > graph->nvtxs/2);

  return graph;
}



/*Malloc cpu 2way-refine params*/
void cuMetis_allocate_cpu_2waymem(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  int nvtxs;
  nvtxs = graph->nvtxs;

  graph->pwgts=(int*)malloc(2*sizeof(int));
  graph->where=(int*)malloc(nvtxs*sizeof(int));
  graph->bndptr=(int*)malloc(nvtxs*sizeof(int));
  graph->bndlist=(int*)malloc(nvtxs*sizeof(int));
  graph->id=(int*)malloc(nvtxs*sizeof(int));
  graph->ed=(int*)malloc(nvtxs*sizeof(int));
}



/*Compute cpu 2way-refine params*/
void cuMetis_compute_cpu_2wayparam(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  int i,j,nvtxs,nbnd,mincut,istart,iend,tid,ted,me;
  int *xadj,*vwgt,*adjncy,*adjwgt,*pwgts;
  int *where,*bndptr,*bndlist,*id,*ed;

  nvtxs= graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;

  pwgts=cuMetis_int_set_value(2,0,graph->pwgts);
  bndptr=cuMetis_int_set_value(nvtxs,-1,graph->bndptr);

  bndlist=graph->bndlist;

  for(i=0;i<nvtxs;i++){
    pwgts[where[i]] += vwgt[i];
  }

  for(nbnd=0,mincut=0,i=0;i<nvtxs;i++){
    istart=xadj[i];
    iend=xadj[i+1];
    me=where[i];
    tid=ted=0;

    for(j=istart;j<iend;j++){
      if(me==where[adjncy[j]]){
        tid+=adjwgt[j];
      }
      else{
        ted+=adjwgt[j];
      }
    }

    id[i]=tid;
    ed[i]=ted;

    if(ted>0||istart==iend){
      cuMetis_listinsert(nbnd,bndlist,bndptr,i);
      mincut+=ted;
    }
  }

  graph->mincut=mincut/2;
  graph->nbnd=nbnd; 

}




/*Compute cpu imbalance params*/
float cuMetis_compute_cpu_imbal(cuMetis_graph_t *graph, int nparts, \
float *part_balance, float *ubvec)
{
  int j,*pwgts;
  float max,cur;
  pwgts=graph->pwgts;
  max=-1.0;

  for(j=0;j<nparts;j++){
    cur=pwgts[j]*part_balance[j]-ubvec[0];

    if(cur>max){
      max=cur;
    }
  }

  return max;
}




/*Init queue */
 void cuMetis_queue_init(cuMetis_queue_t *queue, size_t maxnodes)
{
  int i;
  queue->nnodes=0;
  queue->maxnodes=maxnodes;
  queue->heap=(cuMetis_rkv_t*)malloc(sizeof(cuMetis_rkv_t)*maxnodes);
  queue->locator=(ssize_t*)malloc(sizeof(ssize_t)*maxnodes);

  for(i=0;i<maxnodes;i++){
    queue->locator[i]=-1;
  }

}



/*Create queue*/
cuMetis_queue_t *cuMetis_queue_create(size_t maxnodes)
{
  cuMetis_queue_t *queue; 
  queue = (cuMetis_queue_t *)malloc(sizeof(cuMetis_queue_t));

  cuMetis_queue_init(queue, maxnodes);

  return queue;
}



/*Insert node to queue*/
int cuMetis_queue_insert(cuMetis_queue_t *queue, int node, int key)
{
  ssize_t i,j;
  ssize_t *locator=queue->locator;
  cuMetis_rkv_t *heap=queue->heap;
  i = queue->nnodes++;

  while(i>0){
    j=(i-1)>>1;

    if(M_GT_N(key,heap[j].key)){
      heap[i]=heap[j];
      locator[heap[i].val]=i;
      i=j;
    }
    else
      break;
  }

  heap[i].key=key;
  heap[i].val=node;
  locator[node]=i;

  return 0;

}



/*Get top of queue*/
int cuMetis_queue_top(cuMetis_queue_t *queue)
{
  ssize_t i, j;
  ssize_t *locator;
  cuMetis_rkv_t *heap;

  int vtx, node;
  float key;

  if (queue->nnodes==0){
    return -1;
  }

  queue->nnodes--;
  heap=queue->heap;
  locator=queue->locator;
  vtx=heap[0].val;
  locator[vtx]=-1;

  if ((i=queue->nnodes)>0){
    key=heap[i].key;
    node=heap[i].val;
    i=0;

    while((j=2*i+1)<queue->nnodes){
      if(M_GT_N(heap[j].key,key)){
        if(j+1 < queue->nnodes&&M_GT_N(heap[j+1].key,heap[j].key)){
          j=j+1;
        }

        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else if(j+1<queue->nnodes&&M_GT_N(heap[j+1].key,key)){
        j=j+1;
        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else
        break;
    }

    heap[i].key=key;
    heap[i].val=node;
    locator[node]=i;

  }

  return vtx;

}


/*Delete node of queue*/
int cuMetis_queue_delete(cuMetis_queue_t *queue, int node)
{
  ssize_t i, j, nnodes;
  float newkey, oldkey;
  ssize_t *locator=queue->locator;

  cuMetis_rkv_t *heap=queue->heap;

  i=locator[node];
  locator[node]=-1;

  if(--queue->nnodes>0&&heap[queue->nnodes].val!=node) {
    node=heap[queue->nnodes].val;
    newkey=heap[queue->nnodes].key;
    oldkey=heap[i].key;

    if(M_GT_N(newkey,oldkey)){ 
      while(i>0){
        j=(i-1)>>1;

        if(M_GT_N(newkey,heap[j].key)){
          heap[i]=heap[j];
          locator[heap[i].val]=i;
          i=j;
        }
        else
          break;
      }
    }
    else{ 
      nnodes=queue->nnodes;

      while((j=(i<<1)+1)<nnodes){
        if(M_GT_N(heap[j].key,newkey)){
          if(j+1<nnodes&&M_GT_N(heap[j+1].key,heap[j].key)){
            j++;
          }

          heap[i]=heap[j];
          locator[heap[i].val]=i;
          i=j;
        }
        else if(j+1<nnodes&&M_GT_N(heap[j+1].key,newkey)){
          j++;
          heap[i]=heap[j];
          locator[heap[i].val]=i;
          i=j;
        }
        else
          break;
      }
    }

    heap[i].key=newkey;
    heap[i].val=node;
    locator[node]=i;

  }

  return 0;
}



/*Update queue node key*/
void cuMetis_queue_update(cuMetis_queue_t *queue, int node, int newkey)
{
  ssize_t i, j, nnodes;
  float oldkey;
  ssize_t *locator=queue->locator;

  cuMetis_rkv_t *heap=queue->heap;
  oldkey=heap[locator[node]].key;
  i=locator[node];

  if(M_GT_N(newkey,oldkey)){ 
    while(i>0){
      j=(i-1)>>1;

      if(M_GT_N(newkey,heap[j].key)){
        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else
        break;
    }
  }
  else{ 
    nnodes = queue->nnodes;

    while((j=(i<<1)+1)<nnodes){
      if(M_GT_N(heap[j].key,newkey)){
        if(j+1<nnodes&&M_GT_N(heap[j+1].key,heap[j].key)){
          j++;
        }

        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else if(j+1<nnodes&&M_GT_N(heap[j+1].key,newkey)){
        j++;
        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else
        break;
    }
  }

  heap[i].key=newkey;
  heap[i].val=node;
  locator[node]=i;
  return;

}



/*Free queue*/
void cuMetis_queue_free(cuMetis_queue_t *queue)
{
  if(queue == NULL) return;

  free(queue->heap);
  free(queue->locator);

  queue->maxnodes = 0;

  free(queue);
}



/*Reset queue*/
void cuMetis_queue_reset(cuMetis_queue_t *queue)
{
  ssize_t i;
  ssize_t *locator=queue->locator;

  cuMetis_rkv_t *heap=queue->heap;

  for(i=queue->nnodes-1;i>=0;i--){
    locator[heap[i].val]=-1;
  }

  queue->nnodes=0;

}



/*Balance two partition by moving boundary vertex*/
void cuMetis_bndvertex_2way_bal(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph, float *ntpwgts)
{
  int i,ii,j,k,kwgt,nvtxs,nbnd,nswaps,from,to,temp;
  int *xadj,*vwgt,*adjncy,*adjwgt,*where,*id,*ed,*bndptr,*bndlist,*pwgts;
  int *moved,*perm;

  cuMetis_queue_t *queue;
  int higain,mincut,mindiff;
  int tpwgts[2];

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;
  pwgts=graph->pwgts;
  bndptr=graph->bndptr;
  bndlist=graph->bndlist;

  moved=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  perm=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);

  tpwgts[0]=graph->tvwgt[0]*ntpwgts[0];
  tpwgts[1]=graph->tvwgt[0]-tpwgts[0];
  mindiff=abs(tpwgts[0]-pwgts[0]);
  from=(pwgts[0]<tpwgts[0]?1:0);
  to=(from+1)%2;

  queue=cuMetis_queue_create(nvtxs);
  cuMetis_int_set_value(nvtxs,-1,moved);
  nbnd=graph->nbnd;
  cuMetis_int_randarrayofp(nbnd,perm,nbnd/5,1);

  for(ii=0;ii<nbnd;ii++){
    i=perm[ii];

    if(where[bndlist[i]]==from&&vwgt[bndlist[i]]<=mindiff){
      cuMetis_queue_insert(queue,bndlist[i],ed[bndlist[i]]-id[bndlist[i]]);
    }
  }

  mincut=graph->mincut;

  for(nswaps=0;nswaps<nvtxs;nswaps++) 
  {
    if((higain=cuMetis_queue_top(queue))==-1)
      break;
    if(pwgts[to]+vwgt[higain]>tpwgts[to])
      break;

    mincut-=(ed[higain]-id[higain]);
    cuMetis_add_sub(pwgts[to],pwgts[from],vwgt[higain]);

    where[higain]=to;
    moved[higain]=nswaps;
    cuMetis_swap(id[higain],ed[higain],temp);

    if(ed[higain]==0&&xadj[higain]<xadj[higain+1]){ 
      cuMetis_listdelete(nbnd,bndlist,bndptr,higain);
    }

    for(j=xadj[higain];j<xadj[higain+1];j++){
      k=adjncy[j];
      kwgt=(to==where[k]?adjwgt[j]:-adjwgt[j]);
      cuMetis_add_sub(id[k],ed[k],kwgt);

      if(bndptr[k]!=-1){ 
        if(ed[k]==0){ 
          cuMetis_listdelete(nbnd,bndlist,bndptr,k);

          if(moved[k]==-1&&where[k]==from&&vwgt[k]<=mindiff){ 
            cuMetis_queue_delete(queue,k);
          }
        }
        else{ 
          if(moved[k]==-1&&where[k]==from&&vwgt[k]<=mindiff){
            cuMetis_queue_update(queue,k,ed[k]-id[k]);
          }
        }
      }
      else{
        if(ed[k]>0){  
          cuMetis_listinsert(nbnd,bndlist,bndptr,k);

          if(moved[k]==-1&&where[k]==from&&vwgt[k]<=mindiff){ 
            cuMetis_queue_insert(queue,k,ed[k]-id[k]);
          }
        }
      }
    }
  }

  graph->mincut=mincut;
  graph->nbnd=nbnd;
  cuMetis_queue_free(queue);

}



/*Balance 2-way partition*/
void cuMetis_2way_bal(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph, float *ntpwgts)
{
  if(cuMetis_compute_cpu_imbal(graph,2,cuMetis_admin->part_balance,cuMetis_admin->ubfactors)<=0){ 
    return;
  }

  if(abs(ntpwgts[0]*graph->tvwgt[0]-graph->pwgts[0])<3*graph->tvwgt[0]/graph->nvtxs){
    return;
  }

  cuMetis_bndvertex_2way_bal(cuMetis_admin,graph,ntpwgts);
}



/*Cpu graph refine two partitions*/
void cuMetis_cpu_2way_refine(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph, float *ntpwgts, int iteration_num)
{
  int i,ii,j,k,kwgt,nvtxs,nbnd,nswaps,from,to,pass,limit,temp;
  int *xadj,*vwgt,*adjncy,*adjwgt,*where,*id,*ed,*bndptr,*bndlist,*pwgts;
  int *moved,*swaps,*perm;

  cuMetis_queue_t *queues[2];
  int higain,mincut, mindiff,origdiff,initcut,newcut,mincutorder,avgvwgt;
  int tpwgts[2];

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;
  pwgts=graph->pwgts;
  bndptr=graph->bndptr;
  bndlist=graph->bndlist;

  moved=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  swaps=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  perm=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);

  tpwgts[0]=graph->tvwgt[0]*ntpwgts[0];
  tpwgts[1]=graph->tvwgt[0]-tpwgts[0];

  limit=cuMetis_min(cuMetis_max(0.01*nvtxs,15),100);
  avgvwgt=cuMetis_min((pwgts[0]+pwgts[1])/20,2*(pwgts[0]+pwgts[1])/nvtxs);

  queues[0]=cuMetis_queue_create(nvtxs);
  queues[1]=cuMetis_queue_create(nvtxs);

  origdiff=abs(tpwgts[0]-pwgts[0]);
  cuMetis_int_set_value(nvtxs,-1,moved);

  for(pass=0;pass<iteration_num;pass++){ 
    cuMetis_queue_reset(queues[0]);
    cuMetis_queue_reset(queues[1]);

    mincutorder=-1;
    newcut=mincut=initcut=graph->mincut;
    mindiff=abs(tpwgts[0]-pwgts[0]);
    nbnd=graph->nbnd;
    cuMetis_int_randarrayofp(nbnd,perm,nbnd,1); 

    for(ii=0;ii<nbnd;ii++){
      i=perm[ii];
      cuMetis_queue_insert(queues[where[bndlist[i]]],bndlist[i],ed[bndlist[i]]-id[bndlist[i]]);
    }       

    for(nswaps=0;nswaps<nvtxs;nswaps++){
      from=(tpwgts[0]-pwgts[0]<tpwgts[1]-pwgts[1]?0:1);
      to=(from+1)%2;

      if((higain=cuMetis_queue_top(queues[from]))==-1){
        break;
      }

      newcut-=(ed[higain]-id[higain]);
      cuMetis_add_sub(pwgts[to],pwgts[from],vwgt[higain]);

      if((newcut<mincut&&abs(tpwgts[0]-pwgts[0])<=origdiff+avgvwgt)|| 
          (newcut==mincut&&abs(tpwgts[0]-pwgts[0])<mindiff)){
        mincut=newcut;
        mindiff=abs(tpwgts[0]-pwgts[0]);
        mincutorder=nswaps;
      }
      else if(nswaps-mincutorder>limit){ 
        newcut+=(ed[higain]-id[higain]);
        cuMetis_add_sub(pwgts[from],pwgts[to],vwgt[higain]);
        break;
      }

      where[higain]=to;
      moved[higain]=nswaps;
      swaps[nswaps]=higain;

      cuMetis_swap(id[higain],ed[higain],temp);

      if(ed[higain]==0&&xadj[higain]<xadj[higain+1]){ 
        cuMetis_listdelete(nbnd,bndlist,bndptr,higain);
      }

      for(j=xadj[higain];j<xadj[higain+1];j++){
        k=adjncy[j];
        kwgt=(to==where[k]?adjwgt[j]:-adjwgt[j]);
        cuMetis_add_sub(id[k],ed[k],kwgt);

        if(bndptr[k]!=-1){ 
          if(ed[k]==0){ 
            cuMetis_listdelete(nbnd,bndlist,bndptr,k);
            
            if(moved[k]==-1){  
              cuMetis_queue_delete(queues[where[k]],k);
            }
          }
          else{ 
            if(moved[k]==-1){ 
              cuMetis_queue_update(queues[where[k]],k,ed[k]-id[k]);
            }
          }
        }
        else{
          if(ed[k]>0){  
            cuMetis_listinsert(nbnd,bndlist,bndptr,k);
            
            if(moved[k]==-1){ 
              cuMetis_queue_insert(queues[where[k]],k,ed[k]-id[k]);
            }
          }
        }
      }
    }

    for(i=0;i<nswaps;i++){
      moved[swaps[i]]=-1;  
    }

    for(nswaps--;nswaps>mincutorder;nswaps--){
      higain=swaps[nswaps];
      to=where[higain]=(where[higain]+1)%2;
      cuMetis_swap(id[higain],ed[higain],temp);

      if(ed[higain]==0&&bndptr[higain]!=-1&&xadj[higain]<xadj[higain+1]){
        cuMetis_listdelete(nbnd,bndlist,bndptr,higain);
      }
      else if(ed[higain]>0&&bndptr[higain]==-1){
        cuMetis_listinsert(nbnd,bndlist,bndptr,higain);
      }

      cuMetis_add_sub(pwgts[to],pwgts[(to+1)%2],vwgt[higain]);

      for(j=xadj[higain];j<xadj[higain+1];j++){
        k=adjncy[j];
        kwgt=(to==where[k]?adjwgt[j]:-adjwgt[j]);
        cuMetis_add_sub(id[k],ed[k],kwgt);

        if(bndptr[k]!=-1&&ed[k]==0){
          cuMetis_listdelete(nbnd,bndlist,bndptr,k);
        }
        if(bndptr[k]==-1&&ed[k]>0){
          cuMetis_listinsert(nbnd,bndlist,bndptr,k);
        }
      }
    }

    graph->mincut=mincut;
    graph->nbnd=nbnd;

    // printf("pass=%d nvtxs=%d\n",pass,nvtxs);
    // printf("graph->mincut=%d\n\n",graph->mincut);

    if(mincutorder<=0||mincut==initcut){
      break;
    }

  }

  cuMetis_queue_free(queues[0]);
  cuMetis_queue_free(queues[1]);

}



/*Cpu growbisection algorithm*/
void cuMetis_cpu_growbisection(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, float *ntpwgts, int niparts)
{
  int i,j,k,nvtxs,dd,nleft,first,last,pwgts[2],oneminpwgt,onemaxpwgt, 
      bestcut=0,iter;

  int *xadj,*vwgt,*adjncy,*where;
  int *queue,*tra,*bestwhere;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;

  cuMetis_allocate_cpu_2waymem(cuMetis_admin,graph);

  where=graph->where;

  bestwhere=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  queue=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  tra=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);

  onemaxpwgt=cuMetis_admin->ubfactors[0]*graph->tvwgt[0]*ntpwgts[1];
  oneminpwgt=(1.0/cuMetis_admin->ubfactors[0])*graph->tvwgt[0]*ntpwgts[1]; 
  
  for (iter=0; iter<niparts; iter++){

    cudaDeviceSynchronize();
    gettimeofday(&begin_part_bfs,NULL);
    
    cuMetis_int_set_value(nvtxs,1,where);
    cuMetis_int_set_value(nvtxs,0,tra);

    pwgts[1]=graph->tvwgt[0];
    pwgts[0]=0;
    queue[0]=cuMetis_int_randinrange(nvtxs);
    tra[queue[0]]=1;
    first=0; 
    last=1;
    nleft=nvtxs-1;
    dd=0;

    for(;;){
      if(first==last){ 
        if(nleft==0||dd){
          break;
        }

        k=cuMetis_int_randinrange(nleft);

        for(i=0;i<nvtxs;i++){
          if(tra[i]==0){
            if(k==0){
              break;
            }
            else{
              k--;
            }
          }
        }

        queue[0]=i;
        tra[i]=1;
        first=0; 
        last=1;
        nleft--;
      }

      i=queue[first++];

      if(pwgts[0]>0&&pwgts[1]-vwgt[i]<oneminpwgt){
        dd=1;
        continue;
      }

      where[i]=0;

      cuMetis_add_sub(pwgts[0],pwgts[1],vwgt[i]);

      if(pwgts[1]<=onemaxpwgt){
        break;
      }

      dd=0;

      for(j=xadj[i];j<xadj[i+1];j++){
        k=adjncy[j];

        if(tra[k]==0){
          queue[last++]=k;
          tra[k]=1;
          nleft--;
        }
      }
    }

    cudaDeviceSynchronize();
    gettimeofday(&end_part_bfs,NULL);
    part_bfs += (end_part_bfs.tv_sec - begin_part_bfs.tv_sec) * 1000 + (end_part_bfs.tv_usec - begin_part_bfs.tv_usec) / 1000.0;

    cuMetis_compute_cpu_2wayparam(cuMetis_admin,graph);
    cuMetis_2way_bal(cuMetis_admin,graph,ntpwgts);

    cudaDeviceSynchronize();
    gettimeofday(&begin_part_2refine,NULL);

    cuMetis_cpu_2way_refine(cuMetis_admin,graph,ntpwgts,cuMetis_admin->iteration_num);

    cudaDeviceSynchronize();
    gettimeofday(&end_part_2refine,NULL);
    part_2refine += (end_part_2refine.tv_sec - begin_part_2refine.tv_sec) * 1000 + (end_part_2refine.tv_usec - begin_part_2refine.tv_usec) / 1000.0;
    
    if(iter==0||bestcut>graph->mincut){
      bestcut=graph->mincut;
      cuMetis_int_copy(nvtxs,where,bestwhere);
      
      if(bestcut==0){
        break;
      }
    }
  }

  graph->mincut=bestcut;
  cuMetis_int_copy(nvtxs,bestwhere,where);

}



/*Free graph params*/
void cuMetis_free_graph(cuMetis_graph_t **r_graph) 
{
  cuMetis_graph_t *graph;
  graph=*r_graph;

  free(graph->xadj);
  free(graph->vwgt);
  free(graph->adjncy);
  free(graph->adjwgt);
  free(graph->where);
  free(graph->pwgts);
  free(graph->id);
  free(graph->ed);
  free(graph->bndptr);
  free(graph->bndlist);
  free(graph->tvwgt);
  free(graph->tvwgt_reverse);
  free(graph->label);
  free(graph->cmap);
  free(graph);

  *r_graph = NULL;
}



/*Cpu graph 2-way projection*/
void cuMetis_2way_project(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  int i,j,istart,iend,nvtxs,nbnd,me,tid,ted;
  int *xadj,*adjncy,*adjwgt;
  int *cmap,*where,*bndptr,*bndlist;
  int *cwhere,*cbndptr;
  int *id,*ed;

  cuMetis_graph_t *cgraph;
  cuMetis_allocate_cpu_2waymem(cuMetis_admin,graph);

  cgraph=graph->coarser;
  cwhere=cgraph->where;
  cbndptr=cgraph->bndptr;
  nvtxs=graph->nvtxs;
  cmap=graph->cmap;
  xadj=graph->xadj;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;

  bndptr=cuMetis_int_set_value(nvtxs,-1,graph->bndptr);
  bndlist=graph->bndlist;

  for(i=0;i<nvtxs;i++){
    j=cmap[i];
    where[i]=cwhere[j];
    cmap[i]=cbndptr[j];
  }

  for(nbnd=0,i=0;i<nvtxs;i++){
    istart=xadj[i];
    iend=xadj[i+1];
    tid=ted=0;

    if(cmap[i]==-1){ 
      for(j=istart;j<iend;j++){
        tid+=adjwgt[j];
      }
    }
    else{ 
      me=where[i];

      for(j=istart;j<iend;j++){
        if(me==where[adjncy[j]]){
          tid += adjwgt[j];
        }
        else{
          ted+=adjwgt[j];
        }
      }
    }

    id[i]=tid;
    ed[i]=ted;

    if(ted>0||istart==iend){ 
      cuMetis_listinsert(nbnd,bndlist,bndptr,i);
    }

  }

  graph->mincut=cgraph->mincut;
  graph->nbnd=nbnd;

  cuMetis_int_copy(2,cgraph->pwgts,graph->pwgts);
  cuMetis_free_graph(&graph->coarser);
  graph->coarser=NULL;

}



/*Cpu refinement algorithm*/
void cuMetis_cpu_refinement(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *orggraph, cuMetis_graph_t *graph, float *tpwgts)
{
  cuMetis_compute_cpu_2wayparam(cuMetis_admin,graph);

  for(;;){
    cuMetis_2way_bal(cuMetis_admin,graph,tpwgts);

    cudaDeviceSynchronize();
    gettimeofday(&begin_part_2refine,NULL);

    cuMetis_cpu_2way_refine(cuMetis_admin,graph,tpwgts,cuMetis_admin->iteration_num); 

    cudaDeviceSynchronize();
    gettimeofday(&end_part_2refine,NULL);
    part_2refine += (end_part_2refine.tv_sec - begin_part_2refine.tv_sec) * 1000 + (end_part_2refine.tv_usec - begin_part_2refine.tv_usec) / 1000.0;
    
    if(graph==orggraph){
      break;
    }

    graph=graph->finer;

    cudaDeviceSynchronize();
    gettimeofday(&begin_part_2map,NULL);

    cuMetis_2way_project(cuMetis_admin,graph);

    cudaDeviceSynchronize();
    gettimeofday(&end_part_2map,NULL);
    part_2map += (end_part_2map.tv_sec - begin_part_2map.tv_sec) * 1000 + (end_part_2map.tv_usec - begin_part_2map.tv_usec) / 1000.0;
  }

}


/*Cpu multilevel bisection algorithm*/
int cuMetis_cpu_mlevelbisect(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, float *tpwgts)
{
  int niparts,bestobj=0,curobj=0,*bestwhere=NULL;
  cuMetis_graph_t *cgraph;

  cuMetis_compute_2way_balance(cuMetis_admin,graph,tpwgts);
  cgraph=cuMetis_cpu_coarsen(cuMetis_admin,graph);

  niparts=5;
  cuMetis_cpu_growbisection(cuMetis_admin,cgraph,tpwgts,niparts);

  cuMetis_cpu_refinement(cuMetis_admin,graph,cgraph,tpwgts);
 
  curobj=graph->mincut;
  bestobj=curobj;

  if(bestobj!=curobj){
    cuMetis_int_copy(graph->nvtxs,bestwhere,graph->where);
    cuMetis_compute_cpu_2wayparam(cuMetis_admin,graph);
  }

  return bestobj;
}



/*Set split graph params*/
cuMetis_graph_t *cuMetis_set_splitgraph(cuMetis_graph_t *graph, \
int snvtxs, int snedges)
{
  cuMetis_graph_t *sgraph;
  sgraph=cuMetis_create_cpu_graph();

  sgraph->nvtxs=snvtxs;
  sgraph->nedges=snedges;

  sgraph->xadj=(int*)malloc(sizeof(int)*(snvtxs+1));
  sgraph->vwgt=(int*)malloc(sizeof(int)*(snvtxs+1));
  sgraph->adjncy=(int*)malloc(sizeof(int)*(snedges));
  sgraph->adjwgt=(int*)malloc(sizeof(int)*(snedges));
  sgraph->label=(int*)malloc(sizeof(int)*(snvtxs));
  sgraph->tvwgt=(int*)malloc(sizeof(int));
  sgraph->tvwgt_reverse=(float*)malloc(sizeof(float));

  return sgraph;

}



/*Split graph to lgraph and rgraph*/
void cuMetis_splitgraph(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, cuMetis_graph_t **r_lgraph, cuMetis_graph_t **r_rgraph)
{
  int i,j,k,l,istart,iend,mypart,nvtxs,snvtxs[2],snedges[2];
  int *xadj,*vwgt,*adjncy,*adjwgt,*label,*where,*bndptr;
  int *sxadj[2],*svwgt[2],*sadjncy[2],*sadjwgt[2],*slabel[2];
  int *rename;
  int *temp_adjncy,*temp_adjwgt;

  cuMetis_graph_t *lgraph,*rgraph;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  label=graph->label;
  where=graph->where;
  bndptr=graph->bndptr;

  rename=cuMetis_int_malloc_space(cuMetis_admin,nvtxs);
  snvtxs[0]=snvtxs[1]=snedges[0]=snedges[1]=0;

  for(i=0;i<nvtxs;i++){
    k=where[i];
    rename[i]=snvtxs[k]++;
    snedges[k]+=xadj[i+1]-xadj[i];
  }

  lgraph=cuMetis_set_splitgraph(graph,snvtxs[0],snedges[0]);
  sxadj[0]=lgraph->xadj;
  svwgt[0]=lgraph->vwgt;
  sadjncy[0]=lgraph->adjncy; 	
  sadjwgt[0]=lgraph->adjwgt; 
  slabel[0]=lgraph->label;

  rgraph=cuMetis_set_splitgraph(graph,snvtxs[1],snedges[1]);
  sxadj[1]=rgraph->xadj;
  svwgt[1]=rgraph->vwgt;
  sadjncy[1]=rgraph->adjncy; 	
  sadjwgt[1]=rgraph->adjwgt; 
  slabel[1]=rgraph->label;

  snvtxs[0]=snvtxs[1]=snedges[0]=snedges[1]=0;
  sxadj[0][0]=sxadj[1][0]=0;

  for(i=0;i<nvtxs;i++){
    mypart=where[i];
    istart=xadj[i];
    iend=xadj[i+1];

    if(bndptr[i]==-1){ 
      temp_adjncy=sadjncy[mypart]+snedges[mypart]-istart;
      temp_adjwgt=sadjwgt[mypart]+snedges[mypart]-istart;

      for(j=istart;j<iend;j++){
        temp_adjncy[j]=adjncy[j];
        temp_adjwgt[j]=adjwgt[j]; 
      }

      snedges[mypart]+=iend-istart;
    }
    else{
      temp_adjncy=sadjncy[mypart];
      temp_adjwgt=sadjwgt[mypart];
      l=snedges[mypart];

      for(j=istart;j<iend;j++){
        k=adjncy[j];
        
        if(where[k]==mypart){
          temp_adjncy[l]=k;
          temp_adjwgt[l++]=adjwgt[j]; 
        }
      }
      snedges[mypart]=l;
    }

    svwgt[mypart][snvtxs[mypart]]=vwgt[i];
    slabel[mypart][snvtxs[mypart]]=label[i];
    sxadj[mypart][++snvtxs[mypart]]=snedges[mypart];
  }

  for(mypart=0;mypart<2;mypart++){
    iend=sxadj[mypart][snvtxs[mypart]];
    temp_adjncy=sadjncy[mypart];

    for(i=0;i<iend;i++){ 
      temp_adjncy[i]=rename[temp_adjncy[i]];
    }
  }

  lgraph->nedges=snedges[0];
  rgraph->nedges=snedges[1];

  cuMetis_set_graph_tvwgt(lgraph);
  cuMetis_set_graph_tvwgt(rgraph);

  *r_lgraph=lgraph;
  *r_rgraph=rgraph;

}



/*Cpu Multilevel resursive bisection*/
int cuMetis_mlevel_rbbisection(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, int nparts, int *part, float *tpwgts, int fpart)
{
  int i,nvtxs,objval;
  int *label,*where;

  cuMetis_graph_t *lgraph,*rgraph;
  float wsum,*tpwgts2;

  if(graph->nvtxs==0){
    printf("****You are trying to partition too many parts!****\n");
    return 0;
  }

  nvtxs=graph->nvtxs;

  tpwgts2=cuMetis_float_malloc_space(cuMetis_admin);
  tpwgts2[0]=cuMetis_float_sum((nparts>>1),tpwgts);
  tpwgts2[1]=1.0-tpwgts2[0];

  objval=cuMetis_cpu_mlevelbisect(cuMetis_admin,graph,tpwgts2);
  
  label=graph->label;
  where=graph->where;

  for(i=0;i<nvtxs;i++){
    part[label[i]]=where[i]+fpart;
  }
  for(i=0;i<nvtxs;i++){
    part[label[i]]=where[i]+fpart;
  }

  if(nparts>2){ 
    cudaDeviceSynchronize();
    gettimeofday(&begin_part_slipt,NULL);

    cuMetis_splitgraph(cuMetis_admin,graph,&lgraph,&rgraph);

    cudaDeviceSynchronize();
    gettimeofday(&end_part_slipt,NULL);
    part_slipt += (end_part_slipt.tv_sec - begin_part_slipt.tv_sec) * 1000 + (end_part_slipt.tv_usec - begin_part_slipt.tv_usec) / 1000.0;
  }

  cuMetis_free_graph(&graph);

  wsum=cuMetis_float_sum((nparts>>1),tpwgts);
  
  cuMetis_tpwgts_rescale((nparts>>1),1.0/wsum,tpwgts);
  cuMetis_tpwgts_rescale(nparts-(nparts>>1),1.0/(1.0-wsum),tpwgts+(nparts>>1));
  
  if(nparts>3){
    objval+=cuMetis_mlevel_rbbisection(cuMetis_admin,lgraph,(nparts>>1),part,tpwgts,fpart);
    objval+=cuMetis_mlevel_rbbisection(cuMetis_admin,rgraph,nparts-(nparts>>1),part,tpwgts+(nparts>>1),fpart+(nparts>>1));
  }
  else if(nparts==3){
    cuMetis_free_graph(&lgraph);
    objval+=cuMetis_mlevel_rbbisection(cuMetis_admin,rgraph,nparts-(nparts>>1),part,tpwgts+(nparts>>1),fpart+(nparts>>1));
  }
  
  return objval;

}




/*Set kway balance params*/
void cuMetis_set_kway_bal(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph)
{
  int i,j;

  for(i=0;i<cuMetis_admin->nparts;i++){
    for(j=0;j<1;j++){
      cuMetis_admin->part_balance[i+j]=graph->tvwgt_reverse[j]/cuMetis_admin->tpwgts[i+j];
    }
  }
}



/*Cpu graph partition algorithm*/
int cuMetis_rbbisection(int *nvtxs, int *xadj, int *adjncy, int *vwgt,int *adjwgt, \
int *nparts, float *tpwgts, float *ubvec, int *objval, int *part)
{
  cuMetis_graph_t *graph;
  cuMetis_admin_t *cuMetis_admin;

  cuMetis_admin = cuMetis_set_graph_admin( *nparts, tpwgts, ubvec);

  graph = cuMetis_set_graph(cuMetis_admin, *nvtxs, xadj, adjncy, vwgt, adjwgt);

  cuMetis_allocatespace(cuMetis_admin, graph);           
  
  *objval = cuMetis_mlevel_rbbisection(cuMetis_admin, graph, *nparts, part, cuMetis_admin->tpwgts, 0);
  
  return 1;
 
}



/*CUDA-kway parjection*/
__global__ void projectback(int *where, int *cwhere, int *cmap, int *nvtxs)
{
  int pi;
  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<nvtxs[0]){
    where[pi]=cwhere[cmap[pi]];
  }
}


/*Kway parjection*/
void cuMetis_kway_project(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{       
  int nvtxs=graph->nvtxs;
  cuMetis_graph_t *cgraph; 

  cgraph=graph->coarser;
  
  projectback<<<nvtxs/32+1,32>>>(graph->cuda_where,cgraph->cuda_where,graph->cuda_cmap,graph->cuda_nvtxs);
}


/*Graph initial partition algorithm*/
void cuMetis_initialpartition(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph)
{
  int objval=0;
  int *bestwhere=NULL;
  float *ubvec=NULL;

  graph->where=(int *)malloc(sizeof(int)*graph->nvtxs);

  ubvec=(float*)malloc(sizeof(float));
  ubvec[0]=(float)pow(cuMetis_admin->ubfactors[0],1.0/log(cuMetis_admin->nparts));
  
  cuMetis_rbbisection(&graph->nvtxs,graph->xadj,graph->adjncy,graph->vwgt,graph->adjwgt, \
    &cuMetis_admin->nparts,cuMetis_admin->tpwgts,ubvec,&objval,graph->where);
  
  free(ubvec);
  free(bestwhere);
}



/*CUDA-init pwgts array*/
__global__ void initpwgts(int *cuda_pwgts, int *a)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
    cuda_pwgts[ii]=0;
  }

}


/*CUDA-init pwgts array*/
__global__ void inittpwgts(float *tpwgts, float *temp, int *a)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
    tpwgts[ii]=temp[0];
  }

}



/*Compute sum of pwgts*/
__global__ void Sumpwgts(int *cuda_pwgts, int *cuda_where, int *cuda_vwgt, int *nvtxs)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<nvtxs[0]){
    atomicAdd(&cuda_pwgts[cuda_where[ii]],cuda_vwgt[ii]);
  }
}


/*Malloc initial partition phase to refine phase params*/
void Mallocinit_refineinfo(cuMetis_admin_t *cuMetis_admin,\
cuMetis_graph_t *graph)
{
  int nvtxs=graph->nvtxs;
  int nparts=cuMetis_admin->nparts;

  cudaMalloc((void**)&graph->cuda_where,nvtxs*sizeof(int));
  cudaMemcpy(graph->cuda_where,graph->where,nvtxs*sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_bnd,nvtxs*sizeof(int));

  int num=0;

  cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
  cudaMemcpy(graph->cuda_bndnum,&num,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_nparts,sizeof(int));
  cudaMemcpy(graph->cuda_nparts,&nparts,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_pwgts,nparts*sizeof(int));

  initpwgts<<<nparts/32+1,32>>>(graph->cuda_pwgts,graph->cuda_nparts);

  cudaMalloc((void**)&graph->cuda_nvtxs,sizeof(int));
  cudaMemcpy(graph->cuda_nvtxs,&nvtxs,sizeof(int),cudaMemcpyHostToDevice);

  Sumpwgts<<<nvtxs/32+1,32>>>(graph->cuda_pwgts,graph->cuda_where,graph->cuda_vwgt,graph->cuda_nvtxs);
  
  cudaMalloc((void**)&graph->cuda_tvwgt,sizeof(int));
  cudaMemcpy(graph->cuda_tvwgt,graph->tvwgt,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_tpwgts,nparts*sizeof(float));
  cudaMalloc((void**)&graph->cuda_maxwgt,nparts*sizeof(int));
  cudaMalloc((void**)&graph->cuda_minwgt,nparts*sizeof(int));

  float *temp;
  cudaMalloc((void**)&temp, sizeof(float));
  cudaMemcpy(temp,cuMetis_admin->tpwgts,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_tpwgts,nparts*sizeof(float));

  inittpwgts<<<nparts/32+1,32>>>(graph->cuda_tpwgts,temp,graph->cuda_nparts);

  cudaFree(temp);

}



/*Malloc refine params*/
void cuMetis_malloc_refineinfo (cuMetis_admin_t *cuMetis_admin,\
cuMetis_graph_t *graph)
{
  int nvtxs=graph->nvtxs;
  int nparts=cuMetis_admin->nparts;

  cudaMalloc((void**)&graph->cuda_bnd,nvtxs*sizeof(int));

  int num=0;

  cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
  cudaMemcpy(graph->cuda_bndnum,&num,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_nparts,sizeof(int));
  cudaMemcpy(graph->cuda_nparts,&nparts,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_pwgts,nparts*sizeof(int));

  initpwgts<<<nparts/32+1,32>>>(graph->cuda_pwgts,graph->cuda_nparts);
  
  Sumpwgts<<<nvtxs/32+1,32>>>(graph->cuda_pwgts,graph->cuda_where,graph->cuda_vwgt,graph->cuda_nvtxs);
  
  cudaMalloc((void**)&graph->cuda_tvwgt,sizeof(int));
  cudaMemcpy(graph->cuda_tvwgt,graph->tvwgt,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_tpwgts,nparts*sizeof(float));
  cudaMalloc((void**)&graph->cuda_maxwgt,nparts*sizeof(int));
  cudaMalloc((void**)&graph->cuda_minwgt,nparts*sizeof(int)); 

  float *temp;
  cudaMalloc((void**)&temp, sizeof(float));
  cudaMemcpy(temp,cuMetis_admin->tpwgts,sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&graph->cuda_tpwgts,nparts*sizeof(float));
  inittpwgts<<<nparts/32+1,32>>>(graph->cuda_tpwgts,temp,graph->cuda_nparts);

  cudaFree(temp);

}


/*CUDA-find vertex where ed-id>0 */
__global__ void Find_real_bnd_info(int *cuda_real_bnd_num, int *cuda_real_bnd, int *cuda_where, \
int *cuda_xadj, int *cuda_adjncy, int *cuda_adjwgt, int *cuda_nparts, int *nvtxs)
{
  int pi,me,other,i,me_part;
  pi=blockIdx.x*blockDim.x+threadIdx.x;

  if(pi<nvtxs[0]){
    me=0;
    other=0;
    me_part=cuda_where[pi];

    for(i=cuda_xadj[pi];i<cuda_xadj[pi+1];i++){
      if(cuda_where[cuda_adjncy[i]]==me_part){
        me+=cuda_adjwgt[i];
      }
      else{
        other+=cuda_adjwgt[i];
      }
    }
    if(other>me){
      cuda_real_bnd[atomicAdd(&cuda_real_bnd_num[0],1)]=pi;
    }
  }
}



/*CUDA-find boundary vertex should ro which part*/
__global__ void find_kayparams(int *cuda_real_bnd_num, int *bnd_info, int *cuda_real_bnd, int *cuda_where, \
int *cuda_xadj, int *cuda_adjncy, int *cuda_adjwgt, int *cuda_nparts, int *cuda_bn, int *cuda_bt, int *cuda_g)
{
  int ii,pi,other,i,me_wgt,other_wgt;
  int start,end;

  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<cuda_real_bnd_num[0]){
    pi=cuda_real_bnd[ii];
    start=(cuda_nparts[0])*ii;
    end=(cuda_nparts[0])*(ii+1);

    for(i=start;i<end;i++){
      bnd_info[i]=0;
    }

    for(i=cuda_xadj[pi];i<cuda_xadj[pi+1];i++){
      bnd_info[start+cuda_where[cuda_adjncy[i]]]+=cuda_adjwgt[i];
    }

    me_wgt=other_wgt=bnd_info[start+cuda_where[pi]];

    other=cuda_where[pi];

    for(i=start;i<end;i++){
      if(bnd_info[i]>other_wgt){
        other_wgt=bnd_info[i];
        other=i-start;
      }
    }

    cuda_g[ii]=other_wgt-me_wgt;
    cuda_bt[ii]=other;
    cuda_bn[ii]=pi;

  }
}



/*CUDA-init boundary vertex num*/
__global__ void initbndnum(int *n)
{
  n[0]=0;
}

int refine_pass=1;



/*CUDA-get a csr array*/
__global__ void findcsr(int *bt, int *n, int *nparts, int *bnd_num, int *a)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
    n[2*ii]=-1;
    n[2*ii+1]=-1;

    for(int i=0;i<bnd_num[0];i++){
      if(ii==bt[i]){
        n[2*ii]=i;
        break; 
      }
    }

    if(n[2*ii]!=-1){
      for(int i=n[2*ii];i<bnd_num[0];i++){
        if(bt[i]!=ii){
          n[2*ii+1]=i-1;
          break; 
        }
      }
    }

    n[2*bt[bnd_num[0]-1]+1]=bnd_num[0]-1;

  }
}


/*CUDA-init params*/
__global__ void initcucsr(int *cu_csr,int *bndnum)
{
  cu_csr[0]=0;
  cu_csr[1]=bndnum[0];
}



/*Find boundary vertex information*/
void cuMetis_findgraphbndinfo(cuMetis_admin_t *cuMetis_admin,\
cuMetis_graph_t *graph)
{
  int nvtxs=graph->nvtxs;
  int nparts=cuMetis_admin->nparts;

  initbndnum<<<1,1>>>(graph->cuda_bndnum);

  Find_real_bnd_info<<<nvtxs/32+1,32>>>(graph->cuda_bndnum,graph->cuda_bnd,graph->cuda_where,\
    graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,graph->cuda_nparts,graph->cuda_nvtxs); 
  
  int bnd_num; 
  cudaMemcpy(&bnd_num,graph->cuda_bndnum, sizeof(int), cudaMemcpyDeviceToHost);
  
  if(bnd_num>0){
    cudaMalloc((void**)&graph->cuda_info, bnd_num*nparts* sizeof(int));
    
    find_kayparams<<<bnd_num/32+1,32>>>(graph->cuda_bndnum,graph->cuda_info,graph->cuda_bnd,graph->cuda_where,\
      graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,graph->cuda_nparts,cu_bn,cu_bt,cu_g);
    
    initcucsr<<<1,1>>>(cu_csr,graph->cuda_bndnum);
    
    bb_segsort(cu_bt, cu_bn,bnd_num,cu_csr,  1);
    
    findcsr<<<nparts/32+1,32>>>(cu_bt,cu_que,graph->cuda_nparts,graph->cuda_bndnum,graph->cuda_nparts);
    
    cudaFree(graph->cuda_info);
  }

  graph->cpu_bndnum=(int *)malloc(sizeof(int));
  graph->cpu_bndnum[0]=bnd_num;

}


/*CUDA-get the max/min pwgts*/
__global__ void Sum_maxmin_pwgts(int *cuda_maxwgt, int *cuda_minwgt, float *tpwgts, int *cuda_tvwgt, int *a)
{
  int ii;
  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
    float ubfactor=1.03;

    cuda_maxwgt[ii]=int(tpwgts[ii]*cuda_tvwgt[0]*ubfactor);
    cuda_minwgt[ii]=int(tpwgts[ii]*cuda_tvwgt[0]/ ubfactor);
  }
}



/*CUDA-move vertex*/
__global__ void Exnode_part1(int *que, int *pwgts, int *bndnum, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *minvwgt, int *where, int *a)
{
  int me,to,vvwgt,memax,memin,tomax,tomin;
  int nmoves=0;
  int i,ii;

  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
    if(que[2*ii]!=-1){
      for(i=que[2*ii];i<=que[2*ii+1];i++){
        vvwgt=vwgt[bnd[i]];
        me=where[bnd[i]];
        to=bndto[i];

        memax=maxvwgt[me];
        memin=minvwgt[me];
        tomax=maxvwgt[to];
        tomin=minvwgt[to];

        if(me<=to){
          if(((pwgts[to]+vvwgt>=tomin)&&(pwgts[to]+vvwgt<=tomax))\
          &&((pwgts[me]-vvwgt>=memin)&&(pwgts[me]-vvwgt<=memax))){
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[bnd[i]]=to;
            nmoves++;
          }
        }
      }
    }
  }
}



/*CUDA-move vertex*/
__global__ void Exnode_part2(int *que, int *pwgts, int *bndnum, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *minvwgt, int *where, int *a)
{
  int me,to,vvwgt,memax,memin,tomax,tomin;
  int nmoves=0;
  int i,ii;

  ii=blockIdx.x*blockDim.x+threadIdx.x;

  if(ii<a[0]){
    if(que[2*ii]!=-1){
      for(i=que[2*ii];i<=que[2*ii+1];i++){
        vvwgt=vwgt[bnd[i]];
        me=where[bnd[i]];
        to=bndto[i];

        memax=maxvwgt[me];
        memin=minvwgt[me];
        tomax=maxvwgt[to];
        tomin=minvwgt[to];

        if(me>to){
          if(((pwgts[to]+vvwgt>=tomin)&&(pwgts[to]+vvwgt<=tomax))\
          &&((pwgts[me]-vvwgt>=memin)&&(pwgts[me]-vvwgt<=memax))){    
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[bnd[i]]=to;
            nmoves++;
          }
        }
      }
    }
  }
}




/*Graph multilevel uncoarsening algorithm*/
void cuMetis_uncoarsen(cuMetis_admin_t *cuMetis_admin, cuMetis_graph_t *graph)
{
  int nparts=cuMetis_admin->nparts;

  Sum_maxmin_pwgts<<<nparts/32+1,32>>>(graph->cuda_maxwgt,graph->cuda_minwgt,\
  graph->cuda_tpwgts,graph->cuda_tvwgt,graph->cuda_nparts);

  for(int i=0;i<refine_pass;i++){
    cuMetis_findgraphbndinfo(cuMetis_admin,graph);

    if(graph->cpu_bndnum[0]>0){
      Exnode_part1<<<nparts/32+1,32>>>(cu_que,graph->cuda_pwgts,graph->cuda_bndnum,cu_bn,cu_bt,graph->cuda_vwgt,\
        graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_where,graph->cuda_nparts);  
      
      Exnode_part2<<<nparts/32+1,32>>>(cu_que,graph->cuda_pwgts,graph->cuda_bndnum,cu_bn,cu_bt,graph->cuda_vwgt,\
        graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_where,graph->cuda_nparts);   
    }
    else
    break;

  }
}



/*Free graph uncoarsening phase params*/
void cuMetis_free_uncoarsen(cuMetis_graph_t *graph)
{
  cudaFree(graph->cuda_xadj);
  cudaFree(graph->cuda_cmap);
  cudaFree(graph->cuda_nvtxs);
  cudaFree(graph->cuda_adjncy);
  cudaFree(graph->cuda_adjwgt);
  cudaFree(graph->cuda_vwgt);
  cudaFree(graph->cuda_maxwgt);
  cudaFree(graph->cuda_minwgt);
  cudaFree(graph->cuda_where);
  cudaFree(graph->cuda_pwgts);
  cudaFree(graph->cuda_bnd);
  cudaFree(graph->cuda_bndnum);
  cudaFree(graph->cuda_real_bnd_num);
  cudaFree(graph->cuda_real_bnd);
  cudaFree(graph->cuda_nparts);
  cudaFree(graph->cuda_tvwgt);
  cudaFree(graph->cuda_tpwgts);
}



/*Graph kway-partition algorithm*/
int cuMetis_kway_partition(cuMetis_admin_t *cuMetis_admin, \
cuMetis_graph_t *graph, int *part)
{
  cuMetis_graph_t *cgraph;

  cudaDeviceSynchronize();
  gettimeofday(&begin_part_coarsen,NULL);
  cgraph = cuMetis_coarsen(cuMetis_admin, graph);
  cudaDeviceSynchronize();
  gettimeofday(&end_part_coarsen,NULL);
  part_coarsen += (end_part_coarsen.tv_sec - begin_part_coarsen.tv_sec) * 1000 + (end_part_coarsen.tv_usec - begin_part_coarsen.tv_usec) / 1000.0;
  
  // printf("cnvtxs=%d\n",cgraph->nvtxs);

  cudaDeviceSynchronize();
  gettimeofday(&begin_part_init,NULL);
  cuMetis_initialpartition(cuMetis_admin, cgraph);
  cudaDeviceSynchronize();
  gettimeofday(&end_part_init,NULL);
  part_init += (end_part_init.tv_sec - begin_part_init.tv_sec) * 1000 + (end_part_init.tv_usec - begin_part_init.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_part_uncoarsen,NULL);
  Mallocinit_refineinfo(cuMetis_admin,cgraph);

  cudaDeviceSynchronize();
  gettimeofday(&begin_part_krefine,NULL);

  cuMetis_uncoarsen(cuMetis_admin,cgraph);

  cudaDeviceSynchronize();
  gettimeofday(&end_part_krefine,NULL);
  part_krefine += (end_part_krefine.tv_sec - begin_part_krefine.tv_sec) * 1000 + (end_part_krefine.tv_usec - begin_part_krefine.tv_usec) / 1000.0;
  

  for(int i=0;;i++){
    if(cgraph!=graph){
      cgraph=cgraph->finer;

      cudaMalloc((void**)&cgraph->cuda_where, cgraph->nvtxs*sizeof(int));

      cudaDeviceSynchronize();
      gettimeofday(&begin_part_map,NULL);

      cuMetis_kway_project(cuMetis_admin,cgraph);

      cudaDeviceSynchronize();
      gettimeofday(&end_part_map,NULL);
      part_map += (end_part_map.tv_sec - begin_part_map.tv_sec) * 1000 + (end_part_map.tv_usec - begin_part_map.tv_usec) / 1000.0;

      cuMetis_malloc_refineinfo(cuMetis_admin,cgraph);   

      cudaDeviceSynchronize();
      gettimeofday(&begin_part_krefine,NULL);

      cuMetis_uncoarsen(cuMetis_admin,cgraph);

      cudaDeviceSynchronize();
      gettimeofday(&end_part_krefine,NULL);
      part_krefine += (end_part_krefine.tv_sec - begin_part_krefine.tv_sec) * 1000 + (end_part_krefine.tv_usec - begin_part_krefine.tv_usec) / 1000.0;

      cuMetis_free_uncoarsen(cgraph->coarser);
    } 
    else 
      break; 
  }

  cudaDeviceSynchronize();
  gettimeofday(&end_part_uncoarsen,NULL);
  part_uncoarsen += (end_part_uncoarsen.tv_sec - begin_part_uncoarsen.tv_sec) * 1000 + (end_part_uncoarsen.tv_usec - begin_part_uncoarsen.tv_usec) / 1000.0;
  
  return 0;
}




/*Graph partition algorithm*/
int cuMetis_PartGraph(int *nvtxs,  int *xadj, int *adjncy, int *vwgt,int *adjwgt, \
int *nparts, float *tpwgts, float *ubvec, int *objval, int *part)
{
  cuMetis_graph_t *graph;
  cuMetis_admin_t *cuMetis_admin;

  cuMetis_admin = cuMetis_set_graph_admin( *nparts, tpwgts, ubvec);

  graph = cuMetis_set_graph(cuMetis_admin, *nvtxs, xadj, adjncy, vwgt, adjwgt);

  cuMetis_set_kway_bal(cuMetis_admin, graph);

  cuMetis_admin->Coarsen_threshold = cuMetis_max((*nvtxs)/(20*cuMetis_compute_log2(*nparts)),30*(*nparts));
  
  cuMetis_admin->nIparts=(cuMetis_admin->Coarsen_threshold==30*(*nparts) ?4 :5);
  
  test_time=1;
  
  cuMetis_malloc_original_coarseninfo(cuMetis_admin,graph);  
  
  cudaMalloc((void**)&cu_bn, graph->nvtxs* sizeof(int));
  cudaMalloc((void**)&cu_bt, graph->nvtxs* sizeof(int));
  cudaMalloc((void**)&cu_g, graph->nvtxs*sizeof(int));
  cudaMalloc((void**)&cu_csr, 2*sizeof(int));
  cudaMalloc((void**)&cu_que, 2*cuMetis_admin->nparts*sizeof(int));   
  
  cudaDeviceSynchronize();
  gettimeofday(&begin_part_all,NULL);
  for(int i=0;i<test_time;i++){
    *objval=cuMetis_kway_partition(cuMetis_admin,graph,part);
  }
  cudaDeviceSynchronize();
  gettimeofday(&end_part_all,NULL);
  part_all += (end_part_all.tv_sec - begin_part_all.tv_sec) * 1000 + (end_part_all.tv_usec - begin_part_all.tv_usec) / 1000.0;
  
  cudaMemcpy(part,graph->cuda_where, graph->nvtxs*sizeof(int), cudaMemcpyDeviceToHost);

  cuMetis_free_coarsen(graph);
  cuMetis_free_uncoarsen(graph);

  cudaFree(cu_bn);
  cudaFree(cu_bt);
  cudaFree(cu_g);
  cudaFree(cu_csr);
  cudaFree(cu_que);

  return 1;

}



/*Error exit*/
void cuMetis_error_exit(char *f_str,...)
{
  va_list a;
  va_start(a,f_str);
  vfprintf(stderr,f_str,a);
  va_end(a);

  if (strlen(f_str)==0||f_str[strlen(f_str)-1]!='\n'){
    fprintf(stderr,"\n");
  }

  fflush(stderr);

  if(1)
    exit(-2);
}



/*Open file*/
FILE *cuMetis_fopen(char *fname, char *mode, const char *msg)
{
  FILE *fp;
  char error_message[8192];
  fp=fopen(fname, mode);
  if(fp!=NULL){
    return fp;
  }
  sprintf(error_message,"file: %s, mode: %s, [%s]",fname,mode,msg);
  perror(error_message);
  cuMetis_error_exit("Failed on file fopen()\n");
  return NULL;
}



/*Read graph file*/
cuMetis_graph_t *cuMetis_readgraph(char *filename)
{
  int i,k,fmt,nfields,readew,readvw,readvs,edge,ewgt;
  int *xadj,*adjncy,*vwgt,*adjwgt;
  char *line=NULL,fmtstr[256],*curstr,*newstr;
  size_t lnlen=0;
  FILE *fpin;

  cuMetis_graph_t *graph;
  graph = cuMetis_create_cpu_graph();

  fpin = cuMetis_fopen(filename,"r","Readgraph: Graph");

  do{
    if(getline(&line,&lnlen,fpin)==-1){ 
      cuMetis_error_exit("Premature end of input file: file: %s\n", filename);
    }
  }while(line[0]=='%');

  fmt= 0;
  nfields = sscanf(line, "%d %d %d", &(graph->nvtxs), &(graph->nedges), &fmt);

  if(nfields<2){
    cuMetis_error_exit("The input file does not specify the number of vertices and edges.\n");
  }

  if(graph->nvtxs<=0||graph->nedges<=0){
   cuMetis_error_exit("The supplied nvtxs:%d and nedges:%d must be positive.\n",graph->nvtxs,graph->nedges);
  }

  if(fmt>111){ 
    cuMetis_error_exit("Cannot read this type of file format [fmt=%d]!\n",fmt);
  }

  sprintf(fmtstr,"%03d",fmt%1000);
  readvs=(fmtstr[0]=='1');
  readvw=(fmtstr[1]=='1');
  readew=(fmtstr[2]=='1');

  graph->nedges *=2;

  xadj=graph->xadj=(int*)malloc(sizeof(int)*(graph->nvtxs+1));
  for(i=0;i<graph->nvtxs+1;i++){
    xadj[i]=graph->xadj[i]=0;
  }

  adjncy=graph->adjncy=(int*)malloc(sizeof(int)*(graph->nedges));

  vwgt=graph->vwgt= (int*)malloc(sizeof(int)*(graph->nvtxs));

  for(i=0;i<graph->nvtxs;i++){
    vwgt[i]=graph->vwgt[i]=1;
  }

  adjwgt = graph->adjwgt=(int*)malloc(sizeof(int)*(graph->nedges));
  for(i=0;i<graph->nedges;i++){
    adjwgt[i]=graph->adjwgt[i]=1;
  }

  for(xadj[0]=0,k=0,i=0;i<graph->nvtxs;i++){
    do{
      if(getline(&line,&lnlen,fpin)==-1){
      cuMetis_error_exit("Premature end of input file while reading vertex %d.\n", i+1);
      } 
    }while(line[0]=='%');

    curstr=line;
    newstr=NULL;

    if(readvw){
      vwgt[i]=strtol(curstr, &newstr, 10);

      if(newstr==curstr){
        cuMetis_error_exit("The line for vertex %d does not have enough weights "
          "for the %d constraints.\n", i+1, 1);
      }
      if(vwgt[i]<0){
        cuMetis_error_exit("The weight vertex %d and constraint %d must be >= 0\n", i+1, 0);
      }
      curstr = newstr;
    }

    while(1){
      edge=strtol(curstr,&newstr,10);
      if(newstr==curstr){
        break; 
      }

      curstr=newstr;
      if (edge< 1||edge>graph->nvtxs){
        cuMetis_error_exit("Edge %d for vertex %d is out of bounds\n",edge,i+1);
      }

      ewgt=1;

      if(readew){
        ewgt=strtol(curstr,&newstr,10);

        if(newstr==curstr){
          cuMetis_error_exit("Premature end of line for vertex %d\n", i+1);
        }

        if(ewgt<=0){
          cuMetis_error_exit("The weight (%d) for edge (%d, %d) must be positive.\n",    ewgt, i+1, edge);
        }

        curstr=newstr;
      }

      if(k==graph->nedges){
        cuMetis_error_exit("There are more edges in the file than the %d specified.\n", graph->nedges/2);
      }

      adjncy[k]=edge-1;
      adjwgt[k]=ewgt;
      k++;

    } 
    xadj[i+1]=k;

  }
  fclose(fpin);

  if(k!=graph->nedges){
    printf("------------------------------------------------------------------------------\n");
    printf("***  I detected an error in your input file  ***\n\n");
    printf("In the first line of the file, you specified that the graph contained\n"
      "%d edges. However, I only found %d edges in the file.\n", graph->nedges/2,k/2);
    if(2*k==graph->nedges){
      printf("\n *> I detected that you specified twice the number of edges that you have in\n");
      printf("    the file. Remember that the number of edges specified in the first line\n");
      printf("    counts each edge between vertices v and u only once.\n\n");
    }
    printf("Please specify the correct number of edges in the first line of the file.\n");
    printf("------------------------------------------------------------------------------\n");
    exit(0);
  }
  free(line);
  return graph;
}



/*Write to file*/
void cuMetis_writetofile(char *fname, int *part, int n, int nparts)
{
  FILE *fpout;
  int i;
  char filename[1280000];
  sprintf(filename, "%s.part.%d", fname, nparts);

  fpout = cuMetis_fopen(filename, "w", __func__);

  for (i=0; i<n; i++){
    fprintf(fpout,"%d\n",part[i]);
  }

  fclose(fpout);
}


/*Main function*/
int main(int argc, char **argv)
{  
  cudaSetDevice(1);

  int i;
  int nparts;
  char *filename=(argv[1]);
  nparts =atoi(argv[2]);

  cuMetis_graph_t *graph;

  int *part;
  int objval;
  graph=cuMetis_readgraph(filename); 

  int c;
  c=nparts;
  float tpwgts[c];
  for(i=0;i<c;i++){
    tpwgts[i]=1/c;
  }
  
  float ubvec=1.03;
  part=(int*)malloc(sizeof(int)*(graph->nvtxs));
  
  cuMetis_PartGraph(&graph->nvtxs, graph->xadj, graph->adjncy, graph->vwgt,graph->adjwgt, &nparts, tpwgts, &ubvec,  &objval, part);
  
  printf("Graph: %s \nVertex: %d Edge: %d\n",filename,graph->nvtxs,graph->nedges);

  printf("cuMetis-Partition-end\n");
  printf("cuMetis_Partition_time= %lf     ms\n",part_all);
  printf("------Coarsen_time=          %lf     ms\n",part_coarsen);
  printf("------Init_time=             %lf     ms\n",part_init);
  printf("------Uncoarsen_time=        %lf     ms\n",part_uncoarsen);
  
  //cuMetis_writetofile(filename, part, graph->nvtxs, nparts); 

  printf("------------------------------------------\n");
  printf("The match pattern=           %lf     ms\n",part_match+part_cmatch);
  printf("The multi-node pattern=      %lf     ms\n",part_contract+part_ccontract+part_2map+part_map);
  printf("The 2refine pattern=         %lf     ms\n",part_2refine);
  printf("The krefine pattern=         %lf     ms\n",part_krefine);
  printf("Bfs+Slipt=                   %lf     ms\n",part_bfs+part_slipt);

  int e=cuMetis_computecut(graph, part);
  printf("Edge-cut=                    %d\n",e);
}

