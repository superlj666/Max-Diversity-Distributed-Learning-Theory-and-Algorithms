# NIPS_2019_DC
# Max-Discrepancy Distributed Learning: Fast Risk Bounds and Algorithms
Experiments for the paper "[Max-Discrepancy Distributed Learning: Fast Risk Bounds and Algorithms](https://arxiv.org/abs/1902.04768)", based on a refined version of parameter sever: [ps-lite](https://github.com/dmlc/ps-lite).
## Usage of source code
We implement all methods in C++ and some utils in MATLAB, run scripts are written in Shell as well.
Code used in experiments locates in ./code
### Enviroment
We do experiments based on following softwares:
1. g++ >= 4.8.0
2. MATLAB R2017b
3. ps-lite from https://github.com/dmlc/ps-lite
4. Intel Math Kernel Library 2018
### Data sets
1. All data sets are from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
### Code structure
#### ./include and ./src
1. util.h: some util functions, including efficient matrix computation based on Intel® Math Kernel Library.
2. ridge_regression.h and ridge_regression.cc are used to define a RR class, which is used to implement RR, DRR and MDD-LS.
3. kernel_ridge_regression.h and kernel_ridge_regression.cc are used to define a KRR class, but it is only used to implement KRR.
4. dist_krr.h and dist_krr.cc are used to define a distributed KRR class, which is used to implement KDRR and MDD-RKHS.
#### ./tests
1. test_gaussian_kernel.cc is used to generate gaussian kernel efficiently.
2. test_krr_central.cc is used to test global KRR method.
3. test_krr_dist.cc is used to test KDRR and MDD-RKHS methods, which is implemented based on ps-lite.
4. test_rr_central.cc is used to test global RR method.
5. test_rr_dist.cc is used to test DRR and MDD-LS methods, which is also implemented based on ps-lite.
#### ./scripts
1. feature_map.m is used to mapping feature to [0, 1] and do random shuffle operation to datasets.
2. make.sh is used to complile the code.
3. cross_validation_file.sh is used to split file and mv the to certain dictionary.
4. gaussian_kernel.sh is used to generate different kernels.
5. distributed.sh is used to run distributed methods including DRR, KDRR, MDD-LS and MDD-RKHS.
6. AllInOne.sh is a simple script to run all test.