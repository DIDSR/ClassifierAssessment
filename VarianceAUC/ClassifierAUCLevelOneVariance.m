function [auc,var_auc]=ClassifierAUCLevelOneVariance(h1,h0)
%function [auc,var_auc]=ClassifierAUCLevelOneVariance(h1,h0)
%Estimating the variance of the AUC of a classifier that accounts for the 
%finite size of the testing set but conditional on a particular training set. 
%
%INPUT:
%
%h1: [n1x1]. train the classifier with the training set and test it on the
%     test set, h1 is the testing scores on the actually positive cases
%h0: [n0x1], test scores on the actually negative cases
%    
%OUTPUT:
%
%auc: estimated AUC
%var_auc: variance of AUC that accounts only for the finite size of the
%testing set
%
%Reference:
%W. Chen, B. D. Gallas, W. A. Yousef, "Classifier variability: accounting
%for training and testing," Pattern Recognition Volume 45, Issue 7, July 2012, Pages 2661-2671
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


h1=h1(:);
h0=h0(:);
n1=length(h1);
n0=length(h0);
h1=repmat(h1,1,n0);
h0=repmat(h0',n1,1);
mwk=(h1>h0) + 0.5*(h1==h0); %success matrix
auc=mean(mwk(:));

if nargout==1
    return;
end
    
q1=sum(sum(mwk.^2));
m1=q1/(n0*n1);
q2=sum(sum(mwk,2).^2);
m2=(q2-q1)/(n1*n0*(n0-1));
q3=sum(sum(mwk,1).^2);
m3=(q3-q1)/(n0*n1*(n1-1));
q4=sum(sum(mwk,1))^2;
m4=(q4-q2-q3+q1)/(n0*n1*(n0-1)*(n1-1));
var_auc=m1/(n0*n1)+m2*(n0-1)/(n0*n1)+m3*(n1-1)/(n0*n1)+m4*((n0-1)*(n1-1)/(n0*n1)-1);
