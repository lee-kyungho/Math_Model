clear

data0 = importdata('Porter_data.txt');
data = prepare_railway_data(data0);

Week = data.Week;
Lakes = data.Lakes;
GR = data.GR;
PO = data.PO;
TQG = data.TQG;
DM = data.DM;
SEAS = data.SEAS;
n = data.n;

betahat_first = OLS([ones(n,1) Lakes DM PO SEAS],log(GR));
Yhat = data.ZD*betahat_first;
betahat_second = OLS([ones(n,1) Lakes Yhat SEAS],log(TQG));



X = data.X;
Z = data.Z;
Y = data.Y;
V = (Z'*Z)/n;

beta_2sls = GMM_linear(X,Y,Z,V);
for optimalW = [false true]
   if optimalW       
       u = Y - X*beta_2sls;
       V = 0;
       for i  = 1:n
           ix = 2*i-1:2*i;
           Zu = Z(ix,:)'*u(ix);
           V = V + Zu*Zu';
       end
       beta_optimal = GMM_linear(X,Y,Z,V);
       std_optimal = gmm_variance(optimalW,data);
   else
       std_2sls = gmm_variance(optimalW,data); 
       
   end
end

disp([beta_2sls(1:3), std_2sls(1:3), beta_optimal(1:3), std_optimal(1:3)])



function std = gmm_variance(optimalW,data)

X = data.X;
Y = data.Y;
Z = data.Z;
n = data.n;

V = (Z'*Z)/n;
beta = GMM_linear(X,Y,Z,V);
u = Y - X*beta;
Vhat = 0;
for i  = 1:n
    ix = 2*i-1:2*i;
    Zu = Z(ix,:)'*u(ix);
    Vhat = Vhat + Zu*Zu';
end
Vhat = Vhat/n;
if optimalW
    V = Vhat/n;
end

G = (-Z'*X)/n;
GW = G'/V;
A = GW*G;
B = GW*Vhat*GW';

Avar = ((A\B)/A)/n;
std = sqrt(diag(Avar));

end


function beta = GMM_linear(X,Y,Z,V)
P = (X'*Z)*(V\Z');
beta = (P*X)\(P*Y);
end


function data = prepare_railway_data(data)

X = data.data;

Week = X(:,1);
Lakes = X(:,2);
GR = X(:,3);
PO = X(:,4);
TQG = X(:,5);
DM = X(:,6:9);
SEAS = X(:,10:end);

n = length(Week);

data.Week = Week;
data.Lakes = Lakes;
data.GR = GR;
data.PO = PO;
data.TQG = TQG;
data.DM = DM;
data.SEAS = SEAS;
data.n = n;

data.XD = [ones(n,1) Lakes log(GR) SEAS];
data.XS = [ones(n,1) DM PO log(TQG) SEAS];
data.ZD = [ones(n,1) Lakes DM PO SEAS];
data.ZS = [ones(n,1) DM PO Lakes SEAS];
data.YD = log(TQG);
data.YS = log(GR);

K = size(data.XD,2) + size(data.XS,2);
L = size(data.ZD,2) + size(data.ZS,2);

data.X = zeros(2*n, K);
data.Z = zeros(2*n, L);
data.Y = zeros(2*n,1);
for i = 1:n
   ix = 2*i-1:i*2;
   data.X(ix,:) = blkdiag(data.XD(i,:),data.XS(i,:));
   data.Z(ix,:) = blkdiag(data.ZD(i,:),data.ZS(i,:));
   data.Y(ix) = [data.YD(i);data.YS(i)];
end
end 




function [betahat, std] = OLS(X,Y)

[N,K] = size(X);

betahat = (X'*X)\(X'*Y);

u = Y - X*betahat;

var = u'*u*inv(X'*X)/(N-K);
std = sqrt(diag(var));
end