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

inline bool LoadModelToBinary(const char *filename, float *matrix, int n, int d)
{
    std::ifstream fin(filename, ios::binary);
    fin.read((char *)matrix, (n * d) * sizeof(float));
    fin.close();
    return true;
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

class KernelData
{
  public:
    int row_size;
    int column_size;
    float *kernel;
    explicit KernelData(string file_path, int left_size, int right_size) : row_size(left_size), column_size(right_size)
    {
        kernel = new float[row_size * column_size]();
        LoadModelToBinary(file_path.c_str(), kernel, row_size, column_size);
    }

    ~KernelData()
    {
        delete kernel;
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
            kernel[i * n + j] = exp((kernel[i * n + j] - norms[i] - norms[j]) / (2 * pow(sigma, 2)));
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
            kernel[i * n_right + j] = exp((kernel[i * n_right + j] - norms_left[i] - norms_right[j]) / (2 * pow(sigma, 2)));
        }
    }
    delete norms_left;
    delete norms_right;
}

/*
Define kernel matrix: kernel[m, n]
for i < m :
    read line_left in file_left;
    for j < n:
        read line_right in file_right:
        for cur in d:
            norm+=pow(line_left[i]-line_right[j], 2);
        kernel[i*n+j]=exp(-norm/(2*sigma^2));
*/
inline void HDGaussianKernel(string path_left, string path_right, int left_size, int right_size, int d, float *kernel, float sigma)
{
    std::ifstream file_left(path_left.c_str());
    std::string line_left, buf_left;

    int cursor_left = 0;
    while (std::getline(file_left, line_left))
    {
        std::ifstream file_right(path_right.c_str());
        std::string line_right, buf_right;

        int cursor_right = 0;
        while (std::getline(file_right, line_right))
        {
            float *feature_left = new float[d]();
            std::istringstream in_left(line_left);
            in_left >> buf_left;
            while (in_left >> buf_left)
            {
                auto ss_left = Split(buf_left, ':');
                feature_left[ToInt(ss_left[0]) - 1] = ToFloat(ss_left[1]);
            }

            float *feature_right = new float[d]();
            std::istringstream in_right(line_right);
            in_right >> buf_right;
            while (in_right >> buf_right)
            {
                auto ss_right = Split(buf_right, ':');
                feature_right[ToInt(ss_right[0]) - 1] = ToFloat(ss_right[1]);
            }

            float square_sum = 0;
            for (int i = 0; i < d; ++i)
            {
                square_sum += pow(feature_left[i] - feature_right[i], 2);
            }

            kernel[cursor_left * right_size + cursor_right] = exp(-square_sum / (2 * pow(sigma, 2)));

            delete feature_left;
            delete feature_right;
        }
        cursor_left++;
    }
}

inline void Predict(Dataset &test, float *weight, float *predict)
{
    cout << test.d << " " << test.n << endl;
    cout << "weight: " << endl;
    rr::PrintMatrix(1, test.d, weight);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                test.n, 1, test.d, 1, test.feature, test.d, weight, 1, 0, predict, 1);
    cout << "Print vector of predict:" << endl;
    PrintMatrix(test.n, 1, predict);
}

inline void KernelPredict(KernelData &pre_kernel, float *weight, float *predict)
{
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                pre_kernel.column_size, 1, pre_kernel.row_size, 1, pre_kernel.kernel, pre_kernel.column_size, weight, 1, 0, predict, 1);
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
            predict[i] += weight[j] * exp((kernel[i * n_train + j] - norms_test[i] - norms_train[j]) / (2 * pow(sigma, 2)));
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

inline string ID2string(int id)
{
    string name;
    if (id > 999)
    {
        name += to_string(id);
    }
    else if (id > 99)
    {
        name += "0";
        name += to_string(id);
    }
    else if (id > 9)
    {
        name += "00";
        name += to_string(id);
    }
    else
    {
        name += "000";
        name += to_string(id);
    }
    return name;
}

} // namespace rr

#endif
