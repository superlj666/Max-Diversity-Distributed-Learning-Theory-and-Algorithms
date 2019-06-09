tic();
[y,X]=libsvmread('/home/bd-dev/lijian/201804_NIPS/experiment/data/meta/YearPredictionMSD_train');
n=size(X, 1);
d=size(X, 2);
wr=zeros(d,1);
lambda=1;
gamma=0.01;


c0=inv((X'*X)+lambda*diag(ones(1,d)))*(X'*y);

[y_t,X_t]=libsvmread('/home/bd-dev/lijian/201804_NIPS/experiment/data/YearPredictionMSD/test_00');
sqrt(sum((y_t-X_t*c0).^2)/size(y_t,1))

% b=(X'*y)/n;
% w=c0-gamma*(wr'*c0./b);

toc();