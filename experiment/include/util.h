#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
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
        feature = new float[n * d]();
        label = new float[n]();
    }

    ~Dataset()
    {
        delete feature;
        delete label;
    }
};

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
PrintMatrix(int row, int column, vector<float> data)
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
GetPartA(float *X, int n, int d, float beta, float *A)
{
    for (int i = 0; i < d; ++i)
    {
        A[i * d + i] = 1;
    }

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, d, n, 1 / (float)n, X, d, X, d, beta, A, d);

    // cout << "Print inversion matrix of A:" << endl;
    // PrintMatrix(d, d, A);
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

inline bool SaveModel(const char *filename, float *matrix, int n, int d)
{
    std::ofstream fout(filename);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            fout << matrix[i * d + j] << ' ';
        }
        fout << std::endl;
    }
    fout.close();

    return true;
}

inline bool LoadModel(const char *filename, float *matrix)
{
    std::ifstream input(filename);
    std::string line, buf;

    int cursor = 0;
    while (std::getline(input, line))
    {
        std::istringstream in(line);
        while (in >> buf)
        {
            matrix[cursor++] = ToFloat(buf);
        }
    }
    input.close();

    return true;
}

inline bool SaveModelToBinary(const char *filename, float *matrix, int n, int d)
{
    std::ofstream fout(filename, ios::binary);
    fout.write((char *)matrix, (n * d) * sizeof(float));
    fout.close();
    return true;
}

inline bool LoadModelToBinary(const char *filename, float *matrix, int n, int d)
{
    std::ifstream fin(filename, ios::binary);
    fin.read((char *)matrix, (n * d) * sizeof(float));
    fin.close();
    return true;
}

// K=exp(-\frac{\|x_i, x_j\|}{2\sigma^2})
inline void GaussianKernel(Dataset &data, float *kernel, float sigma)
{
    int n = data.n;
    int d = data.d;

    float *norms = new float[n]();

    // square sum of rows
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            norms[i] += pow(data.feature[i * d + j], 2);
        }
    }

    // -\|x_i-x_j\|
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, n, d, 2, data.feature, d, data.feature, d, 0, kernel, n);

    // K=exp(-\frac{\|x_i, x_j\|}{2\sigma^2})
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            kernel[i * n + j] = exp((kernel[i * n + j] - norms[i] - norms[j]) / (2 * pow(sigma, 2))) / n;
        }
    }
    delete norms;
}

// K=exp(-\frac{\|x_i, x_j\|}{2\sigma^2})
inline void GaussianKernel(Dataset &left, Dataset &right, float *kernel, float sigma)
{
    int n_left = left.n;
    int d = left.d;
    int n_right = right.n;

    float *norms_left = new float[n_left]();
    float *norms_right = new float[n_right]();
    // square sum of rows
    for (size_t j = 0; j < d; ++j)
    {
        for (size_t i = 0; i < n_left; ++i)
        {
            norms_left[i] += pow(left.feature[i * d + j], 2);
        }
        for (size_t i = 0; i < n_right; ++i)
        {
            norms_right[i] += pow(right.feature[i * d + j], 2);
        }
    }
    // -\|x_i-x_j\|
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_left, n_right, d, 2, left.feature, d, right.feature, d, 0, kernel, n_right);

    // K=exp(-\frac{\|x_i, x_j\|}{2\sigma^2})
    for (size_t i = 0; i < n_left; ++i)
    {
        for (size_t j = 0; j < n_right; ++j)
        {
            kernel[i * n_right + j] = exp((kernel[i * n_right + j] - norms_left[i] - norms_right[j])/ (2 * pow(sigma, 2)));
        }
    }
    delete norms_left;
    delete norms_right;
}

inline void Predict(Dataset &test, float *weight, float *predict)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                test.n, 1, test.d, 1, test.feature, test.d, weight, 1, 0, predict, 1);
    // cout << "Print vector of predict:" << endl;
    // PrintMatrix(test.n, 1, result);
}

inline void KRRPredict(Dataset &train, Dataset &test, float *weight, float *predict, float sigma)
{
    int n_train = train.n;
    int d = train.d;
    int n_test = test.n;

    float *norms_train = new float[n_train]();
    float *norms_test = new float[n_test]();
    float *kernel = new float[n_train * n_test]();

    // square sum of rows
    for (size_t j = 0; j < d; ++j)
    {
        for (size_t i = 0; i < n_train; ++i)
        {
            norms_train[i] += pow(train.feature[i * d + j], 2);
        }
        for (size_t i = 0; i < n_test; ++i)
        {
            norms_test[i] += pow(test.feature[i * d + j], 2);
        }
    }

    // -\|x_i-x_j\|
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_test, n_train, d, 2, test.feature, d, train.feature, d, 0, kernel, n_train);

    // K=exp(-\frac{\|x_i, x_j\|}{2\sigma^2})
    for (size_t i = 0; i < n_test; ++i)
    {
        for (size_t j = 0; j < n_train; ++j)
        {
            predict[i] += weight[j]*exp((kernel[i * n_train + j] - norms_test[i] - norms_train[j])/ (2 * pow(sigma, 2)));
        }
    }
    delete norms_train;
    delete norms_test;
    delete kernel;
}

inline float MSE(Dataset &test, float *predict)
{
    float mse = 0;
    for (int i = 0; i < test.n; ++i)
    {
        mse += pow(predict[i] - test.label[i], 2);
    }
    return sqrt(mse / (float)test.n);
}

} // namespace rr

#endif
