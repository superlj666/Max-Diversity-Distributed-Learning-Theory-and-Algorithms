tic()
% feature mapping to [-1, 1]
file_name='cadata' %'cadata'%''%'abalone
raw_path=['/home/bd-dev/lijian/201804_NIPS/experiment/data/meta/', file_name];
for i=1:10
save_path=strcat('/home/bd-dev/lijian/201804_NIPS/experiment/data/test/meta/', file_name, num2str(i));
[y,instance_matrix]=libsvmread(raw_path);

% shuffle
ran_sort = randperm(size(y,1));
y = y(ran_sort, :);
instance_matrix = instance_matrix(ran_sort, :);

% only use 100000 data for training
y=y-min(y);
max_feature=max(max(instance_matrix));
min_feature=min(min(instance_matrix));

instance_matrix=(instance_matrix-min_feature)/(max_feature-min_feature);
libsvmwrite(save_path, y, sparse(instance_matrix));
end
toc()