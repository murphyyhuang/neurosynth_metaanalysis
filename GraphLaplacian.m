clear; close all; clc;

%% Load Required Data

diff_path = '/home/libilab2/a/users/huan1282/dev/mldata/gnn_rfmri_intermediate_20190919/structure_conn/groupavg_structure_conn_ttest_thres1e-10.mat';
W = load(diff_path);
W = W.subj_edge_avg;

% Number of regions
n_ROI = size(W,1);

%% 2. Symmetric Normalization of adjacency matrix
D=diag(sum(W,2)); %degree
Wsymm=D^(-1/2)*W*D^(-1/2);
Wnew=Wsymm;

%% compute normalized Laplacian
L=eye(n_ROI)-Wnew;

%% Laplacian Decomposition
[U,LambdaL] = eig(L);
[LambdaL, IndL]=sort(diag(LambdaL));
U=U(:,IndL);

%% Compute weighted zero crossings for Laplacian eigenvectors (Supplementary Figure S1)
for u=1:360 %for each eigenvector
    UU=U(:,u);%-mean(U(:,u));
    summ=0;
    for i=1:359 %for each connection
        for j=i+1:360
            if (UU(i)*UU(j))<0 %if signals are of opposite signs
                summ=summ+(W(i,j)>0.06);%W(i,j);
            end
            wZC(u)=summ;
        end
    end
end

figure;plot(wZC);title ('Supplementary Fig. S1');xlabel('Connectome harmonics');ylabel('Weighted zero crossings')

%% Load functional connectivity data
% rs_path = '/home/libilab2/a/users/huan1282/dev/mldata/gnn_rfmri_intermediate_20190919/func_conn/hcp_s1200_test126LR_Glasser360_conn.mat';
% X_RS=load(rs_path);
% X_RS=X_RS.conn_mats;
% X_RS = cat(1, X_RS{1}, X_RS{2});


rs_path = '/home/libilab2/a/users/huan1282/dev/mldata/gnn_rfmri_intermediate_20190919/pretrain_node_feature/graph_test_node_representation_1570842848.mat';
X_RS=load(rs_path);
X_RS = cat(1, X_RS.sess_1, X_RS.sess_2);


X_RS = permute(X_RS, [2 3 1]);

zX_RS = zscore(X_RS, 0, 2);

% number of subjects
nsubjs_RS=size(zX_RS,3);

for s=1:nsubjs_RS
    X_hat_L(:,:,s)=U'*zX_RS(:,:,s);
end

pow=abs(X_hat_L).^2;
PSD=squeeze(mean(pow,2)); % average across different functional connectivity map

avg=mean(PSD')';
stdPSD=std(PSD')';
upper1=avg+stdPSD;
lower1=avg-stdPSD;
idx = max(PSD')>0 & min(PSD')>0 & mean(PSD')>0;         

figure;
patch([LambdaL(idx)', fliplr(LambdaL(idx)')], [lower1(idx)'  fliplr(upper1(idx)')], [0.8 0.8 0.8]);hold on;plot(LambdaL,avg);xlim([0.05 2]);ylim([0.02 50]);title('Supplementary Fig. S2');xlabel('Harmonic Frequency');ylabel('Energy')
set(gca, 'XScale', 'log', 'YScale','log')


%% compute cut-off frequency
mPSD=mean(PSD,2);
AUCTOT=trapz(mPSD(1:360)); %total area under the curve

i=0;
AUC=0;
while AUC<AUCTOT/2
    AUC=trapz(mPSD(1:i));
    i=i+1;
end
NN=i-1; %CUTOFF FREQUENCY C : number of low frequency eigenvalues to consider in order to have the same energy as the high freq ones
NNL=360-NN; 

%% split structural harmonics in high/low frequency

M=fliplr(U); %Laplacian eigenvectors flipped in order (high frequencies first)

Vlow=zeros(size(M));
Vhigh=zeros(size(M));
Vhigh(:,1:NNL)=M(:,1:NNL);%high frequencies= decoupled 
Vlow(:,end-NN+1:end)=M(:,end-NN+1:end);%low frequencies = coupled 


% %% Load graph pretraining intermediate data
% pretrained_path = '/home/libilab2/a/users/huan1282/dev/mldata/gnn_rfmri_intermediate_20190919/pretrain_node_feature/graph_test_node_representation_1570842848.mat';
% X_Pretrain=load(pretrained_path);
% X_Pretrain = cat(1, X_Pretrain.sess_1, X_Pretrain.sess_2);
% zX_Pretrain = zscore(X_Pretrain, 0, 3);
% 
% % number of subjects
% nsubjs_Pretrain=size(zX_Pretrain, 1);
% 
% for s=1:nsubjs_Pretrain
%     X_hat_Pretrain(s, :, :)=U'*squeeze(zX_Pretrain(s, :, :));
% end
% 
% pow_pre=abs(X_hat_Pretrain).^2;
% PSD_pre=squeeze(mean(pow_pre,3)); % average across different functional connectivity map
% 
% avg_pre=mean(PSD_pre)';
% stdPSD_pre=std(PSD_pre)';
% upper1_pre=avg_pre+stdPSD_pre;
% lower1_pre=avg_pre-stdPSD_pre;
% idx = max(PSD_pre)>0 & min(PSD_pre)>0 & mean(PSD_pre)>0;
% 
% figure;
% patch([LambdaL(idx)', fliplr(LambdaL(idx)')], [lower1_pre(idx)'  fliplr(upper1_pre(idx)')], [0.8 0.8 0.8]);hold on;plot(LambdaL,avg_pre);xlim([0.05 2]);ylim([0.02 50]);title('Supplementary Fig. S2');xlabel('Harmonic Frequency');ylabel('Energy')
% set(gca, 'XScale', 'log', 'YScale','log')

