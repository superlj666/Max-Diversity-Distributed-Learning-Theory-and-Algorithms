tic()
file_name= 'YearPredictionMSD' %'abalone'
raw_path=['/home/bd-dev/lijian/201804_NIPS/experiment/data/feature/', file_name];
save_path=['/home/bd-dev/lijian/201804_NIPS/experiment/data/kernel_2/', file_name];
sigma=2;
max_size=512*1024*1024;

[y,instance_matrix]=libsvmread(raw_path);
norms = sum(instance_matrix'.^2);
sample_n = size(instance_matrix,1);
row_size=size(instance_matrix,1);


if file_name == 'YearPredictionMSD'
     step = 5000;
%     times = 500;
%     for i=1:times
% left=instance_matrix((i-1)*step+1:i*step,:);
left=instance_matrix(1:5000,:);
norms_left=sum(left'.^2);
K = exp((-norms_left'*ones(1,sample_n) - ones(step,1)*norms + 2*(left*instance_matrix'))/(2*sigma^2));
toc()
csvwrite(strcat(save_path,'_', string(i)), K);
%         break
%     end
else 
    norms = sum(instance_matrix'.^2);
    K = exp((-norms'*ones(1,sample_n) - ones(sample_n,1)*norms + 2*(instance_matrix*instance_matrix'))/(2*sigma^2));
    dlmwrite([save_path,i], K);
end


