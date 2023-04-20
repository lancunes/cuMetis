#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<sys/time.h>
#include<immintrin.h>
#include"include/tranpose.h"
#include<malloc.h>

#include "include/mmio_highlevel.h"

void merge_findsize(int *startA, int lenA, int *startB, int lenB, int *lenC)
{
    int ptra = 0;
    int ptrb = 0;
    int len = 0;
    while (ptra < lenA && ptrb < lenB)
    {
        int a = startA[ptra];
        int b = startB[ptrb];
        if (a < b)
        {
           // printf("a<b: a = %i, b = %i\n", a, b);
            len++;
            ptra++;
        }
        else if (a > b)
        {
           // printf("a>b: a = %i, b = %i\n", a, b);
            len++;
            ptrb++;
        }
        else if (a == b)
        {
           // printf("a=b: a = %i, b = %i\n", a, b);
            len++;
            ptra++;
            ptrb++;
        }
       // printf("1 ptra = %i, ptrb = %i, len = %i\n", ptra, ptrb, len);
    };

    while (ptra < lenA)
    {
        len++;
        ptra++;
       // printf("2 ptra = %i, ptrb = %i, len = %i\n", ptra, ptrb, len);
    };

    while (ptrb < lenB)
    {
        len++;
        ptrb++;
       // printf("3 ptra = %i, ptrb = %i, len = %i\n", ptra, ptrb, len);
    };

    *lenC = len;
}


void merge_computeval(int *startA, int lenA, int *startB, int lenB, int *startC)
{
    int ptra = 0;
    int ptrb = 0;
    int len = 0;
    while (ptra < lenA && ptrb < lenB)
    {
        int a = startA[ptra];
        int b = startB[ptrb];
        if (a < b)
        {
          //  printf("a<b: a = %i, b = %i\n", a, b);
            startC[len] = a;
            len++;
            ptra++;
        }
        else if (a > b)
        {
           // printf("a>b: a = %i, b = %i\n", a, b);
            startC[len] = b;
            len++;
            ptrb++;
        }
        else if (a == b)
        {
          //  printf("a=b: a = %i, b = %i\n", a, b);
            startC[len] = a;
            len++;
            ptra++;
            ptrb++;
        }
      //  printf("1 ptra = %i, ptrb = %i, len = %i\n", ptra, ptrb, len);
    };

    while (ptra < lenA)
    {
        int a = startA[ptra];
        startC[len] = a;
        len++;
        ptra++;
      //  printf("2 ptra = %i, ptrb = %i, len = %i\n", ptra, ptrb, len);
    };

    while (ptrb < lenB)
    {
        int b = startB[ptrb];
        startC[len] = b;
        len++;
        ptrb++;
      //  printf("3 ptra = %i, ptrb = %i, len = %i\n", ptra, ptrb, len);
    };
}

int main(int argc, char ** argv)
{
    
     char *filename=argv[1];
     int n,nnz,issymmetricr;
     
     int length=0;
     for(int i=0;;i++){
       if(filename[i]!='\0')
       length++;
       else
       break;
     }
     
     
   //  int length=sizeof(filename)/sizeof(char);
     
     char *file1=(char*)malloc(sizeof(char)*(length+2));
     //printf("length=%d\n",length);
     for(int i=0;i<length+2;i++){
       if(i<length-4){
       file1[i]=filename[i];
       }
       else if(i==length-4){file1[i]='.';}
       else if(i==length-3){file1[i]='g';}
       else if(i==length-2){file1[i]='r';}
       else if(i==length-1){file1[i]='a';}
       else if(i==length){file1[i]='p';}
       else if(i==length+1){file1[i]='h';}

     }
    // freopen("Transport.txt","w",stdout);
     
    // printf("%s",file1);
     freopen(file1,"w",stdout);
    //
/*
    char *partitionfile=filename;
    char s2[8]={'.','g','r','a','p','h'};
    int i1=0,j1=0;
    for( i1=0;partitionfile[i1]!='\0';i1++ );
    do
    {
       partitionfile[i1++]=s2[j1];
    } while(s2[j1++]!='\0');
    //printf("partitionfile=%s\n", partitionfile);   
     freopen(partitionfile,"w",stdout);*/
//
    // malloc A in CSR
    mmio_info(&n,&n,&nnz,&issymmetricr,filename);
    int *csrRowPtrA = (int *)malloc(sizeof(int) * (n+1));
    int *csrColIdxA = (int *)malloc(sizeof(int) * nnz);
    float *csrValA = (float *)malloc(sizeof(float) * nnz);
    mmio_data(csrRowPtrA,csrColIdxA,csrValA,filename);


    // malloc A in CSC
    int *cscColPtrA = (int *)malloc(sizeof(int) * (n+1));
    int *cscRowIdxA = (int *)malloc(sizeof(int) * nnz);
    float *cscValA = (float *)malloc(sizeof(float) * nnz);
    matrix_transposition(n, n, nnz, csrRowPtrA, csrColIdxA, csrValA,
                         cscRowIdxA, cscColPtrA, cscValA);

  /*  for (int i = 0; i < n+1; i++)
        printf("cscColPtrA[%i] = %i\n", i, cscColPtrA[i]);
    for (int i = 0; i < nnz; i++)
        printf("cscRowIdxA[%i] = %i\n", i, cscRowIdxA[i]);*/

    // malloc AT in CSR
    int *csrRowPtrAT = cscColPtrA;
    int *csrColIdxAT = cscRowIdxA;
    float *csrValAT = cscValA;

    // malloc C = A+AT
    int *csrRowPtrC = (int *)malloc(sizeof(int) * (n+1));

    // add A and AT in CSR (only for sizes)
    for (int i = 0; i < n; i++)
    {
        int lenC = 0;
        int *startA = &csrColIdxA[csrRowPtrA[i]];
        int lenA = csrRowPtrA[i+1] - csrRowPtrA[i];
        int *startAT = &csrColIdxAT[csrRowPtrAT[i]];
        int lenAT = csrRowPtrAT[i+1] - csrRowPtrAT[i];
        merge_findsize(startA, lenA, startAT, lenAT, &lenC);
        csrRowPtrC[i] = lenC;
    }

    // prefix scan csrRowPtrC
    exclusive_scan(csrRowPtrC, n + 1);

   /* for (int i = 0; i < n+1; i++)
        printf("cscColPtrC[%i] = %i\n", i, csrRowPtrC[i]);*/

    int nnzC = csrRowPtrC[n];
    int *csrColIdxC = (int *)malloc(sizeof(int) * nnzC);
    float *csrValC = (float *)malloc(sizeof(float) * nnzC);

    // add A and AT in CSR (for values)
    for (int i = 0; i < n; i++)
    {
        int lenC = 0;
        int *startA = &csrColIdxA[csrRowPtrA[i]];
        int lenA = csrRowPtrA[i+1] - csrRowPtrA[i];
        int *startAT = &csrColIdxAT[csrRowPtrAT[i]];
        int lenAT = csrRowPtrAT[i+1] - csrRowPtrAT[i];
        int *startC = &csrColIdxC[csrRowPtrC[i]];

        merge_computeval(startA, lenA, startAT, lenAT, startC);
    }

   /* for (int i = 0; i < nnzC; i++)
        printf("csrColIdxC[%i] = %i\n", i, csrColIdxC[i]);*/

    // malloc C = A+AT without diagonal
    int *csrRowPtrCnew = (int *)malloc(sizeof(int) * (n+1));

    // remove diagonals
    for (int i = 0; i < n; i++)
    {
        int len = csrRowPtrC[i+1] - csrRowPtrC[i];
        for (int j = csrRowPtrC[i]; j < csrRowPtrC[i+1]; j++)
            if (csrColIdxC[j] == i)
                len--;
        csrRowPtrCnew[i] = len;
    }

    // prefix scan csrRowPtrC
    exclusive_scan(csrRowPtrCnew, n + 1);

  /*  for (int i = 0; i < n+1; i++)
        printf("cscColPtrCnew[%i] = %i\n", i, csrRowPtrCnew[i]);*/
    for (int i = 0; i < n+1; i++)
       // printf("cscColPtrCnew[%i] = %i\n", i, csrRowPtrCnew[i]);
    {}
    printf("%d\t%i\n", n, csrRowPtrCnew[n]/2);

    int nnzCnew = csrRowPtrCnew[n];
    int *csrColIdxCnew = (int *)malloc(sizeof(int) * nnzCnew);
    float *csrValCnew = (float *)malloc(sizeof(float) * nnzCnew);

    // copy C into Cnew (without dia)
    for (int i = 0; i < n; i++)
    {
        int jnew = csrRowPtrCnew[i];
        for (int j = csrRowPtrC[i]; j < csrRowPtrC[i+1]; j++)
        {
            if (csrColIdxC[j] != i)
            {
                csrColIdxCnew[jnew] = csrColIdxC[j];
                jnew++;
            }
        }
    }

    int a=0;
    while ( a < n+1)
    { 
       
        for(int j=csrRowPtrCnew[a];j<csrRowPtrCnew[a+1];j++)
        {
        //printf("csrColIdxCnew[%i] = %i\t", i, csrColIdxCnew[i]);
           printf("%d  ",csrColIdxCnew[j]+1);
           //a++;
        }
         a++;
        printf("\n");
    } 
    fclose(stdout);
}


























