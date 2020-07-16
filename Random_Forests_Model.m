% Clearing previous command history.
clear; close all; clc

% Ensuring randomness remains constant.
rng('default')

% Loading the data.
data = readtable('liver_disease_clean_data.csv');

% Normalizing the numerical data types.
data = normalize(data,'DataVariables',{'Age','Total_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Total_Protiens','Albumin_and_Globulin_Ratio'});

% Turning category fields into categorical fields.
data.Female = categorical(data.Female);
data.Liver_Disease = categorical(data.Liver_Disease);

% Splitting the data using randperm.
split_size = 0.3;
N = size(data,1);
split = false(N,1);
split(1:round(split_size*N)) = true;
split = split(randperm(N));

trainingSet = data(~split,:);
testingSet = data(split,:);

% Separated variable predictors and output values for training and test set.
trainingPredictors = trainingSet(:,1:7);
trainingOutcomes = trainingSet{:,8};
testingPredictors = testingSet(:,1:7);
testingOutcomes = testingSet{:,8};

X = trainingPredictors;
Y = trainingOutcomes;

% Uncomment the section below to view our bayesian optimization model visualisation.
% Estimated feasible point varies each time the model is run, however, through
% multiple trials, we have found the best estimated hyperparameters with
% the highest accuracy scores and true positive rate (according to the
% model).

%{
% Selecting hyperparameters as number of trees and number of leafs:
minLS = optimizableVariable('minLS',[1,30],'Type','integer');
numT = optimizableVariable('numT',[1,200],'Type','integer');
hyperparametersRF = [minLS;numT];
fun = @(hyp)f(hyp,X,Y);
results = bayesopt(fun,hyperparametersRF);
besthyperparameters = bestPoint(results);
%}

%%%%%%%%%%%%%%%%% Best estimated feasible point (according to models):
   
%%%%%%%%%%%%%%%%%  minLS    numT 
%%%%%%%%%%%%%%%%%  _____    ____ 

%%%%%%%%%%%%%%%%%   22       122   

% Optimization completed.
% MaxObjectiveEvaluations of 30 reached.      
% Total function evaluations: 30
% Total elapsed time: 50.3455 seconds.
% Estimated objective function value = 0.30573
% Estimated function evaluation time = 0.38151

A = testingPredictors;
B = testingOutcomes;

% Optimised Variables from previous tests...
leafOpt = 22;
treeOpt = 122;

opts=statset('UseParallel',true);
optimisedModel = TreeBagger(treeOpt,X,Y,'method','classification','OOBPrediction','on','Options',opts,...
    'MinLeafSize',leafOpt);

% Predicting against test set using optimised model.
[predictedLabels,scores] = predict(optimisedModel,A);

% ROC Curve.
[X,Y,T,AUC] = perfcurve(B,scores(:,1),'1','NegClass','2');
figure
plot(X,Y);
xlabel('False Positive Rate') 
ylabel('True Positive Rate')
title('ROC for Classification by Random Forest')

% Confusion Matrix.
confusionMatrixTest = confusionmat(B,categorical(predictedLabels));
figure
confusionchart(confusionMatrixTest)

% Accuracy.
compareTest = B == predictedLabels;
accuracy = nnz(compareTest) / numel(B) * 100;

% Out-of-Bag Classification Error.
figure
plot(oobError(optimisedModel))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')

%Calculating the accuracy
Accuracymodel = 100*sum(diag(confusionMatrixTest))./sum(confusionMatrixTest(:))
 
%Calculating the recall
Recall = confusionMatrixTest(1,1)/(confusionMatrixTest(1,1)+confusionMatrixTest(1,2))
 
%Calculating the precision
Precision = confusionMatrixTest(1,1)/(confusionMatrixTest(1,1)+confusionMatrixTest(2,1))
 
%Calculating the specificity
Specificity = confusionMatrixTest(2,2)/(confusionMatrixTest(2,1)+confusionMatrixTest(2,2))

% Bayesian Optimization Function.
function oob = f(hyperparameters, X, Y)
opts=statset('UseParallel',true);
A=TreeBagger(hyperparameters.numT,X,Y,'method','classification','OOBPrediction','on','Options',opts,...
    'MinLeafSize',hyperparameters.minLS);
oob = oobError(A, 'Mode','ensemble');
end
% Reference for the function from MathWorks Community: 
% https://uk.mathworks.com/matlabcentral/answers/347547-using-bayesopt-for-treebagger-classification