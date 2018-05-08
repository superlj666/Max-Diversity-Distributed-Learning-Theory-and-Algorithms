#ifndef RR_UTIL_H_
#define RR_UTIL_H_

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "mkl.h"
using namespace std;

namespace rr
{

inline int ToInt(const char *str)
{
    int flag = 1, ret = 0;
    const char *p = str;

    if (*p == '-')
    {
        ++p;
        flag = -1;
    }
    else if (*p == '+')
    {
        ++p;
    }

    while (*p)
    {
        ret = ret * 10 + (*p - '0');
        ++p;
    }
    return flag * ret;
}

inline int ToInt(const std::string &str)
{
    return ToInt(str.c_str());
}

inline float ToFloat(const char *str)
{
    float integer = 0, decimal = 0;
    float base = 1;
    const char *p = str;

    while (*p)
    {
        if (*p == '.')
        {
            base = 0.1;
            ++p;
            continue;
        }
        if (base >= 1.0)
        {
            integer = integer * 10 + (*p - '0');
        }
        else
        {
            decimal += base * (*p - '0');
            base *= 0.1;
        }
        ++p;
    }

    return integer + decimal;
}

inline float ToFloat(const std::string &str)
{
    return ToFloat(str.c_str());
}

inline std::vector<std::string>
Split(std::string line, char sparator)
{
    std::vector<std::string> ret;

    int start = 0;
    std::size_t pos = line.find(sparator, start);
    while (pos != std::string::npos)
    {
        ret.push_back(line.substr(start, pos));
        start = pos + 1;
        pos = line.find(sparator, start);
    }
    ret.push_back(line.substr(start));
    return ret;
}

class Dataset
{
  public:
    int n;
    int d;
    float *feature;
    float *label;
    explicit Dataset(int sample_size, int feature_size) : n(sample_size), d(feature_size)
    {
        feature = (float *)mkl_malloc(n * d * sizeof(float), 64);
        label = (float *)mkl_malloc(n * 1 * sizeof(float), 64);
    }
};

inline void
GetPartA(float *X, int n, int d, float beta, float *A)
{
    for (int i = 0; i < d; ++i)
    {
        A[i * d + i] = 1;
    }

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, d, n, 1 / (float)n, X, d, X, d, beta, A, d);
}

inline bool
MatrixInversion(float *A, int dim)
{
    int ipiv[dim];
    if (!LAPACKE_sgetrf(LAPACK_ROW_MAJOR, dim, dim, A, dim, ipiv))
    {
        return !LAPACKE_sgetri(LAPACK_ROW_MAJOR, dim, A, dim, ipiv);
    }
    return false;
}

inline void
GetPartb(float *X, float *y, int n, int d, float *b)
{
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, 1, n, 1 / (float)n, X, d, y, 1, 0, b, 1);
}

inline void
PrintMatrix(int row, int column, float *data)
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            cout << data[i * column + j] << " ";
        }
        cout << endl;
    }
}

inline void
LoadData(string filename, Dataset &data)
{
    std::ifstream input(filename.c_str());
    std::string line, buf;

    int cursor = 0;
    while (std::getline(input, line))
    {
        std::istringstream in(line);
        in >> buf;
        data.label[cursor] = ToFloat(buf);
        while (in >> buf)
        {
            auto ss = Split(buf, ':');
            data.feature[cursor * data.d + ToInt(ss[0]) - 1] = ToFloat(ss[1]);
        }
        cursor++;
    }
}

inline 
bool SaveModel(std::string &filename, int d)
{
    std::ofstream fout(filename.c_str());
    fout << d << std::endl;
    for (int i = 0; i < d; ++i)
    {
        fout << w_[i] << ' ';
    }
    fout << std::endl;
    fout.close();
    return true;
}
}

#endif
