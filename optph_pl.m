function [ph]=optph_pl(cohmat,ph0)
% Usage: phase linking algorithm
%
% Email response by Stefano Tebaldini: 
% the key to implement the phase linking algorithm is to assume that all 
% phases are known except one. The estimation of this one phase can then be obtained 
% in closed form. So, for the first step of the algorithm I assume that the phases 
% are those that you read on the first column of the covariance matrix (that is, 
% the phase of the interfergrams w.r.t. a common master), then I start estimating the 
% phase in all passes assuming that the others are correct, and I update thier 
% value iteration by iteration.Usually, I see the algorithm converging in some 10 iterations 
% (sometimes I try with 100, but it's likely to be a waste of time).
%
% Note: coherence matrix is used for weighting to get more stable results:
% i.e. cohmat, not pinv(abs(cohmat)).*cohmat
%
%--------input--------%
% cohmat(D,D):complex coherence matrix
% ph0(D,1): initial phase vector
%
%-------output-------%
% ph(D,1): estimated optimal interf phase vector
%
% References:
% A. Monti Guarnieri and S. Tebaldini, "On the Exploitation of Target Statistics for 
% SAR Interferometry Applications," IEEE Transactions on Geoscience & Remote Sensing, 
% vol. 46, pp. 3436 - 3443, 2008.
%
%   ===============================================================
%   31-Aug-2016 10:44:08   by Dong Jie, Song Huina
%   ===============================================================


D=size(cohmat,1);
maxk=100;   % maximum of iterations
epsilon=0.001;
k=1;

ph0=exp(1i*ph0); 
ph=zeros(size(ph0));
[f0]=abs(ph0'*cohmat*ph0);
while( k < maxk) 
    
    for i=1:D
        ph(i)=sum(reshape(cohmat(i,:),D,1).*ph0)-cohmat(i,i)*ph0(i); % complex phase
    end
    ph=ph./abs(ph);
    [f1]=abs(ph'*cohmat*ph);
   
    if abs(f1-f0) < epsilon
        break;
    end  
 
    ph0=ph;     f0=f1;
    k=k+1;
end

ph=ph.*conj(ph(1)); 
ph=angle(ph);

end


