#include <string>  
#include <cstring>
#include <iostream>
#include "mkl.h"
using namespace std;
/* Auxiliary routine: printing a matrix */  
void PrintMatrix(lapack_int m, lapack_int n, double* a)  
{  
    lapack_int i, j;  
    for( i = 0; i < m; i++ )  
    {  
        for( j = 0; j < n; j++ )  {
            cout << a[i*n+j] << endl;
        }
    }  
}  

bool IsFullRank(double* eigVal, int n){
    for (int i=0; i<n; ++i) {
        if (eigVal[i]==0) {
            return false;
        }
    }
    return true;
}

// 计算矩阵的特征值和特征向量
int SEigen(double* pEigVal, double* pEigVec, const double* pSrc, int dim)
{
    double* eigValImag = new double[dim];
    double* eigVecVl = new double[dim * dim];
    double* pSrcBak = new double[dim * dim];
    memcpy(pSrcBak, pSrc, sizeof(double) * dim * dim);

    int nRetVal = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', dim, pSrcBak, dim,
        pEigVal, eigValImag, eigVecVl, dim, pEigVec, dim);   // 计算特征值和特征向量

    delete[] eigValImag;
    eigValImag = nullptr;
    delete[] eigVecVl;
    eigVecVl = nullptr;
    delete[] pSrcBak;
    pSrcBak = nullptr;

    return nRetVal;
}


// 计算实对称矩阵的特征值和特征向量
int SRealSymEigen1(double* pEigVal, double* pEigVec, const double* pSrc, int dim)
{
    memcpy(pEigVec, pSrc, sizeof(double)* dim * dim);
    return LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', dim, pEigVec, dim, pEigVal);   // 计算特征值和特征向量
}

// 计算实对称矩阵的特征值和特征向量
int SRealSymEigen2(double* pEigVal, double* pEigVec, const double* pSrc, int dim)
{
    memcpy(pEigVec, pSrc, sizeof(double)* dim * dim);
    return LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', dim, pEigVec, dim, pEigVal);  // 计算特征值和特征向量
}

int main(int argc,char** argv)  
{  
    const int nDim = 3;
    double pa[nDim * nDim] = { 1, 2, 3, 6, 5, 4, 7, 5, 8 };
    double eigVal[nDim];
    double eigVec[nDim * nDim];

    memset(eigVal, 0, sizeof(double)* nDim);
    memset(eigVec, 0, sizeof(double)* nDim * nDim);
    int ret1 = SEigen(eigVal, eigVec, pa, nDim);  // 计算非对称矩阵的特征值和特征向量
    cout << IsFullRank(eigVal, nDim) << endl;

    memset(eigVal, 0, sizeof(double)* nDim);
    memset(eigVec, 0, sizeof(double)* nDim * nDim);
    int ret2 = SRealSymEigen1(eigVal, eigVec, pa, nDim);  // 计算实对称矩阵的特征值和特征向量

    int ret3 = SRealSymEigen2(eigVal, eigVec, pa, nDim);  // 计算实对称矩阵的特征值和特征向量
    return 0;  
}  
