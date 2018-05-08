#include <string>  
#include "mkl.h"
#include <iostream>
using namespace std;
/* Auxiliary routine: printing a matrix */  
void PrintMatrix(lapack_int m, lapack_int n, double* a)  
{  
    lapack_int i, j;  
    for( i = 0; i < m; i++ )  
    {  
        for( j = 0; j < n; j++ )  {
            cout << a[i*n+j] << " ";
        }
        cout << endl;
    }  
}  

bool MatrixInversion(double* A, int dim) {
    int ipiv[dim];
    if (!LAPACKE_dgetrf(LAPACK_ROW_MAJOR,dim,dim,A,dim,ipiv)) {
        return !LAPACKE_dgetri(LAPACK_ROW_MAJOR,dim,A,dim,ipiv); 
    }
    return false;
}

int main(int argc,char** argv)  
{  
    double a[] =   
    {  
        3,-1,-1,  
        4,-2,-1,  
        -3,2,1  
    };  

    double aT[] =   
    {  
        3,-1,-1,  
        4,-2,-1,  
        -3,2,1  
    };  
    
    double b[]=
    {
        1,0,
        0,2
    };

    cout << "is Full Rank: " << MatrixInversion(a, 3) << endl;
    cout << "Inversion of Matrix a:" << endl;
    PrintMatrix(3,3,a);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    3, 3, 3, 1, a, 3, aT, 3, 0, a, 3);
    cout << "Multiplication of a and a':" << endl;
    PrintMatrix(3,3,a);

    cout << "is Full Rank: " << MatrixInversion(b, 2) << endl;
    cout << "Inversion of Matrix b:" << endl;
    PrintMatrix(2,2,b);

    return 0;  
}  
