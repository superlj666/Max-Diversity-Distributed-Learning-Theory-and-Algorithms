tic()
[y,instance_matrix]=libsvmread('/home/bd-dev/lijian/201804_NIPS/experiment/data/feature/abalone1');
sigma=-2;
y=y(1:3000,:)
instance_matrix=instance_matrix(1:3000,:)
sample_n = size(instance_matrix,1);
norms = sum(instance_matrix'.^2);
K = exp((-norms'*ones(1,sample_n) - ones(sample_n,1)*norms + 2*(instance_matrix*instance_matrix'))/(2*sigma^2));
w=inv(K+0*diag(ones(1,size(y,1))))*y;

[y_test,X_test]=libsvmread('/home/bd-dev/lijian/201804_NIPS/experiment/data/feature/abalone1');
y_test=y_test(3001:4177,:)
X_test=X_test(3001:4177,:)
sample_p = size(X_test,1);
norms_p = sum(X_test'.^2);
K_p = exp((-norms'*ones(1,sample_p) - ones(sample_n,1)*norms_p + 2*(instance_matrix*X_test'))/(2*sigma^2));
predict = K_p'*w

sqrt(sum((predict-y_test).^2)/sample_p)
toc()