#include<stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdarg.h>
#include<time.h>
#include<cuda_runtime.h>
#include<sys/time.h>
#include<thrust/scan.h>
#include<thrust/sort.h>
#include<thrust/execution_policy.h>
#include<thrust/device_ptr.h>
#include "bb_segsort.h"


int cuMetis_PartGraph(int *nvtxs,  int *xadj, int *adjncy, int *vwgt,int *adjwgt, \
int *nparts, float *tpwgts, float *ubvec, int *objval, int *part);
