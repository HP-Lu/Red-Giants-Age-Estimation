clc; clear; close all;

[data0,str0] = xlsread('RG_spectra_Age.xlsm');
[a0,b0] = size(data0);

F_train = data0(:,[1,2,3,4,5])';
A_train = data0(:,15)';

[FF_train,ps_input] = mapstd(F_train);
[AA_train,ps_output] = mapstd(A_train);

RMSE = zeros(100,1);
RE = zeros(100,1);
R_square = zeros(100,1);

fid=fopen('result.txt','a');
fprintf(fid,'%20s %20s %20s %20s \n','k_ID','RMSE','RE','R_square');
fclose(fid); 

for k = 1:100

net = newff(FF_train,AA_train,[16 16]);

net.trainParam.epochs = 1000;
net.trainParam.goal = 0.001;
net.trainParam.lr = 0.01;

[net,tr] = train(net,FF_train,AA_train);

save(['E:\LHP_KC\net_', num2str(k),'.mat'], 'net');
save(['E:\LHP_KC\tr_', num2str(k),'.mat'], 'tr');
 
Age_Outputs = net(FF_train);

AAge_Outputs = mapstd('reverse',Age_Outputs,ps_output);

trOut = Age_Outputs(tr.trainInd);
ttrOut = mapstd('reverse',trOut,ps_output);

vOut = Age_Outputs(tr.valInd);
vvOut = mapstd('reverse',vOut,ps_output);

tsOut = Age_Outputs(tr.testInd);
ttsOut = mapstd('reverse',tsOut,ps_output);

trTarg = A_train(tr.trainInd);
vTarg = A_train(tr.valInd);
tsTarg = A_train(tr.testInd);

RTr = corrcoef(trTarg, ttrOut);
RV = corrcoef(vTarg, vvOut);
RTe = corrcoef(tsTarg, ttsOut);

% R square: 
R_square(k) = 1 - (sum((tsTarg - ttsOut).^2))/(sum((tsTarg - mean(tsTarg)).^2));

%RMSE£¨Root Mean Squarde Error£©: 
RMSE(k) = sqrt(sum((tsTarg - ttsOut).^2)/length(tsTarg));

%RE£¨Relative Error£©£º
RE(k) = mean(abs(tsTarg - ttsOut)./tsTarg);

fid=fopen('result.txt','a');
fprintf(fid,'%20d %20.5f %20.5f %20.5f \n',k,RMSE(k),RE(k),R_square(k));
fclose(fid); 

x = -3:0.1:15;
y = x;

plot(tsTarg, ttsOut,'bo',x,y,'r--')
xlim([-2 14])
ylim([-2 14])
xlabel('Age')
ylabel('Predicted Age')
title('d')
text(-1,13,'Input parameters: 1,2,3,4,5','Color','black','FontSize',12)
string = {['R^2 = ' num2str(R_square(k)),', ','RE = ' num2str(RE(k)),', ','RMSE = ' num2str(RMSE(k))]};
text(-1,12,string,'Color','black','FontSize',11)

print ('-depsc', ['E:\LHP_KC\', num2str(k),'.eps'])

end

RMSE_max = max(RMSE); RMSE_min = min(RMSE); RMSE_mean = mean(RMSE);
RE_max = max(RE); RE_min = min(RE); RE_mean = mean(RE);
R_square_max = max(R_square); R_square_min = min(R_square); R_square_mean = mean(R_square);

fid=fopen('result.txt','a');
fprintf(fid,'\n%20s %20s %20s %20s\n','Name','Max','Min','Mean');
fprintf(fid,'%20s %20.5f %20.5f %20.5f\n','RMSE',RMSE_max,RMSE_min,RMSE_mean);
fprintf(fid,'%20s %20.5f %20.5f %20.5f\n','RE',RE_max,RE_min,RE_mean);
fprintf(fid,'%20s %20.5f %20.5f %20.5f\n','R_square',R_square_max,R_square_min,R_square_mean);
fclose(fid); 
















