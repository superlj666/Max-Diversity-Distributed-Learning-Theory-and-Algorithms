[y,X]=libsvmread('/home/bd-dev/lijian/201804_NIPS/experiment/data/feature/a9a1');

threthold=20000;%size(y,1)*2/3;

train_X=X(1:threthold,:);
train_y=y(1:threthold,:);
test_X=X(threthold+1:size(y,1),:);
test_y=y(threthold+1:size(y,1),:);
n=size(train_y,1);
d=size(train_X,2);

c0=inv((train_X'*train_X)/n+0.01*diag(ones(1,d)))*(train_X'*train_y/n);

label=test_X*c0

sqrt(sum((label-test_y).^2)/size(test_y,1))
