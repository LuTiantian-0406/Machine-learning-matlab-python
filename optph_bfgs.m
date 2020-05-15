function [ph]=optph_bfgs(cohmat,ph0)
% Usage: BFGS: nonlinear quasi-Newton to get optimal values 
%
%--------input--------%
% cohmat(D,D):complex coherence matrix
% ph0(D,1): initial phase vector
%
%-------output-------%
% ph(D,1): estimated optimal interf phase vector
%
% References:
% A. Ferretti, A. Fumagalli, F. Novali, C. Prati, F. Rocca, and A. Rucci, 
% "A new algorithm for processing interferometric data-stacks: SqueeSAR," 
% Geoscience and Remote Sensing, IEEE Transactions on, vol. 49, 
% pp. 3460-3470, 2011.
%
%   ===============================================================
%   31-Aug-2016 10:44:08   by Dong Jie, Song Huina
%   ===============================================================

cohcoh=pinv(abs(cohmat)).*cohmat;

maxFunEvals = 25;
options = [];
options.display = 'none';
options.maxFunEvals = maxFunEvals;
options.maxIter=500;
options.Method = 'lbfgs';
MFUN = @(input) MyFun_optph(input,cohcoh);
% fprintf('Result after %d evaluations of limited-memory solvers on 2D rosenbrock:\n',maxFunEvals);
% fprintf('minFunc with %s \n',options.Method);
[ph,f,exitflag,output] = minFunc(MFUN,ph0,options);

end

