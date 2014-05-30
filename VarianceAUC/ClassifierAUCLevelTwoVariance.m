function [auc,var_total,var_train,var_test,m]=ClassifierAUCLevelTwoVariance(h1c,h0c,h1,h0)
%function [auc,var_total,var_train,var_test,m]=ClassifierAUCLevelTwoVariance(h1c,h0c,h1,h0)
%Estimating the variance of AUC of a classifier that accounts for training and testing 
%This assessment program is designed for the situation where there is one
%dataset for training the classifier and another independent dataset (size n1, n0) for
%testing.
%
%INPUT:
%
%h1c: [n1x1],train the classifier with the training set and test it on the
%     test set, h1c is the testing scores on the actually positive cases
%h0c: [n0x1], test scores on the actually negative cases
%h1: [n1xB]. Bootstrap the training set B times to generate B training
%    sets. Train the classifier with each of the bootstrap training set 
%    and test it on the test set. The bth column of h1 is the testing scores 
%    of the actually positive cases for the classifier trained with the bth training set.
%    The jth row is the testing scores for the jth actually positive case.
%    
%h0: [n0xB], similarly, testing scores on the actually negative cases
%
%OUTPUT:
%
%auc: estimated mean AUC
%var_total: variance of AUC that accounts for both training and testing
%var_train: training variance component
%var_test: testing variance component
%m: moment vector 
%
%Reference:
%W. Chen, B. D. Gallas, W. A. Yousef, "Classifier variability: accounting
%for training and testing," submitted to IEEE Trans PAMI
%
%Weijie Chen 03/30/2011
%
%@@@@@@@@@@@@@@Disclaimer@@@@@@@@@@@@@@@@@@@
%This software and documentation (the "Software") were developed at the 
%Food and Drug Administration (FDA) by employees of the Federal Government 
%in the course of their official duties. Pursuant to Title 17, Section 105 
%of the United States Code, this work is not subject to copyright protection 
%and is in the public domain. Permission is hereby granted, free of charge,
%to any person obtaining a copy of the Software, to deal in the Software 
%without restriction, including without limitation the rights to use, copy,
%modify, merge, publish, distribute, sublicense, or sell copies of the 
%Software or derivatives, and to permit persons to whom the Software is 
%furnished to do so. FDA assumes no responsibility whatsoever for use by 
%other parties of the Software, its source code, documentation or compiled 
%executables, and makes no guarantees, expressed or implied, about its 
%quality, reliability, or any other characteristic. Further, use of this 
%code in no way implies endorsement by the FDA or confers any advantage in 
%regulatory decisions. Although this software can be redistributed and/or 
%modified freely, we ask that any derivative works bear some notice that 
%they are derived from it, and any modified versions bear some notice that 
%they have been modified.
%@@@@@@@@@@@@@@End of Disclaimer@@@@@@@@@@@@@@@@@@@

 
%calculate the auc
h1c=h1c(:);
h0c=h0c(:);
n1=length(h1c);
n0=length(h0c);
h1c=repmat(h1c,1,n0);
h0c=repmat(h0c',n1,1);
mwk=(h1c>h0c) + 0.5*(h1c==h0c); %success matrix
auc=mean(mwk(:));

[n0,B]=size(h0); [n1,B1]=size(h1);
if B ~=B1
    error('The number of bootstraps for h0 and h1 must be the same.');
end

%now assessing the variance
s=zeros(n0,n1,B);%success matrix
for b=1:B
    x0=h0(:,b);   x0=repmat(x0,1,n1);
    x1=h1(:,b);   x1=repmat(x1',n0,1);
    s(:,:,b)=(x1>x0) + 0.5*(x1==x0);
end

m1=s.^2; 
m1=mean(m1(:));
m2=sum(sum(sum(s).^2))/(n0^2*n1*B);
m3=sum(sum(sum(s,2).^2))/(n0*n1^2*B);
m4=sum(sum(sum(s)).^2)/(n0^2*n1^2*B);
m5=sum(sum(sum(s,3).^2))/(n0*n1*B^2);
m6=sum(sum(sum(s),3).^2)/(n0^2*n1*B^2);
m7=sum(sum(sum(s,2),3).^2)/(n0*n1^2*B^2);
m8=sum(sum(sum(s)))^2/(n0^2*n1^2*B^2);
trxB=[1            0              0                   0 
    1/n0      (n0-1)/n0           0                   0 
    1/n1           0         (n1-1)/n1                0 
    1/(n0*n1) (n0-1)/(n0*n1) (n1-1)/(n0*n1) (n1-1)*(n0-1)/(n0*n1)];
trxB=[trxB zeros(4)
    trxB/B (B-1)*trxB/B];
m=inv(trxB)*[m1;m2;m3;m4;m5;m6;m7;m8];
c1=1/(n0*n1);      c2=(n0-1)/(n0*n1);
c3=(n1-1)/(n0*n1); c4=(n0-1)*(n1-1)/(n0*n1);
var_total=c1*m(1)+c2*m(2)+c3*m(3)+c4*m(4)-m(8);
var_train=m(4)-m(8);
var_test=c1*m(1)+c2*m(2)+c3*m(3)-(1-c4)*m(4);