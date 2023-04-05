 clc
 clear all
 close all
 fpath='E:\';
 %data = fullfile(fpath,'trainnn');
 tdata=imageDatastore(fpath,'includesubfolders',true,'LabelSource','foldername');
 count=tdata.countEachLabel;
  [tdata valdat]= splitEachLabel(tdata,0.80, 'randomized');
 %Load Pre Traind Model
% if true
 % tdata = augmentedImageDatastore([227,227],tdata,'ColorPreprocessing','gray2rgb');
%end
%count=data.countEachLabel;
net=alexnet;
layers=[imageInputLayer([227 227 3])
    net.Layers(2:end-3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer()
    ]
options=trainingOptions('adam','Maxepoch',5,'InitialLearnRate',0.001);
xxrr=trainNetwork(tdata,layers,options)

%analyzeNetwork(xxrr)

allclass=[];
allscore=[];
for i=1:length(valdat.Labels)
a=readimage(valdat,i);
[outclass, score]=classify(xxrr,a);
allclass=[allclass outclass];
allscore=[allscore score];
end
%confusion matrix val
result=horzcat(valdat.Labels,allclass')
% predicted=allclass;
figure
plotconfusion(valdat.Labels,predicted');
 accuracy = mean(allclass==valdat.Labels)