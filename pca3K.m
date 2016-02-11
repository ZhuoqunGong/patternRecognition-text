clear;

%%%%%%%%%%%%%%%% process training imgs %%%%%%%%%%%%%%%%%%%%%%%%%

data=load('train_3k_mnist.mat');
imgs = data.imgs;
labels=data.labels;

newImgs=[];
for i=(1:3000)
img= imgs(:,:,i);
newImg=reshape(img,[400,1]);
newImgs=[newImgs,newImg];
end

[V,D]=eig(newImgs*newImgs');
newV=V(:,1:30);
features=[];

for i=(1:3000)
feature=newV\newImgs(:,i);
features=[features,feature];
end
featuresT=features';
%%to fit the SVM

%%%%%%%%%%%%%%%%%%%%%%%%%%% train %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = svmtrain(labels, featuresT, '-c 1 -g 0.07 -b 1');

%%%%%%%%%%%%%%%%%%%%%%%%%%% process test imgs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testData=load('test_10k_mnist.mat');
testImgs=testData.imgs;
testLabels=testData.labels;

newTestImgs=[];
for i=(1:10000)
testImg= testImgs(:,:,i);
newTestImg=reshape(testImg,[400,1]);
newTestImgs=[newTestImgs,newTestImg];
end

[testV,testD]=eig(newTestImgs*newTestImgs');
newTestV=testV(:,1:30);
testFeatures=[];

for i=(1:10000)
testFeature=newTestV\newTestImgs(:,i);
testFeatures=[testFeatures,testFeature];
end
testFeaturesT=testFeatures';
%to fit the SVM

%%%%%%%%%% test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[predict_label, accuracy, prob_values] = svmpredict(testLabels, testFeaturesT, model, '-b 1');

