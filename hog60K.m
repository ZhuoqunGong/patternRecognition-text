clear;

%%%%%%%%%%%%%%%% process training imgs %%%%%%%%%%%%%%%%%%%%%%%%%

data=load('train_60k_mnist.mat');
imgs = data.imgs;
labels=data.labels;

features=[];
for i=(1:60000)
feature=double(extractHOGFeatures(imgs(:,:,i)));
features=[features;feature];
end

selectedFeatures=features(:,1:36);

%featuresT=selectedFeatures';
%%to fit the SVM

%%%%%%%%%%%%%%%%%%%%%%%%%%% train %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = svmtrain(labels, selectedFeatures, '-c 1 -g 0.07 -b 1');

%%%%%%%%%%%%%%%%%%%%%%%%%%% process test imgs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testData=load('test_10k_mnist.mat');
testImgs=testData.imgs;
testLabels=testData.labels;

testFeatures=[];
for i=(1:10000)
testFeature=double(extractHOGFeatures(testImgs(:,:,i)));
testFeatures=[testFeatures;testFeature];
end
selectedTestFeatures=testFeatures(:,1:36);



%testFeaturesT=selectedTestFeatures';
%to fit the SVM

%%%%%%%%%% test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[predict_label, accuracy, prob_values] = svmpredict(testLabels, selectedTestFeatures, model, '-b 1');

