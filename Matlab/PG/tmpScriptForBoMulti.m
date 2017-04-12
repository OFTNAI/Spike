experimentName = '6.3grayAndBlack_a--StatDec_FFLATFB_20EP_testMulti_FB5_rand123';
layer = 4;
FRThreshold = 200.0;
nBins = 5;

%%load FR
trainedNet = 0;
FR_u = loadFR(experimentName,layer,trainedNet);
FR_u(FR_u>FRThreshold) = FRThreshold;

trainedNet = 1;
FR_t = loadFR(experimentName,layer,trainedNet);
FR_t(FR_t>FRThreshold) = FRThreshold;




% %% convert FR struct to BO
% FR_u_BO = zeros(4,2,64*64);
% FR_u_BO(1,:,:) = FR_u(1,1:2,:);
% FR_u_BO(2,:,:) = FR_u(1,3:4,:);
% FR_u_BO(3,:,:) = FR_u(2,1:2,:);
% FR_u_BO(4,:,:) = FR_u(2,3:4,:);
% 
% FR_t_BO = zeros(4,2,64*64);
% FR_t_BO(1,:,:) = FR_t(1,1:2,:);
% FR_t_BO(2,:,:) = FR_t(1,3:4,:);
% FR_t_BO(3,:,:) = FR_t(2,1:2,:);
% FR_t_BO(4,:,:) = FR_t(2,3:4,:);

% running InfoAnalysis
%IRs_u = infoAnalysis(FR_u,0,nBins);
IRs_t = infoAnalysis(FR_t,1,nBins);

% %% find BO selective cells
% minBound = FRThreshold*0.8;
% maxBound = FRThreshold*0.2;
% obj1selective = [];
% obj2selective = [];
% obj3selective = [];
% obj4selective = [];
% for i = 1:size(FR_t_BO,3)
%     if (min(FR_t_BO(1,:,i))>minBound && max(max(FR_t_BO([2,3,4],:,i)))<maxBound)
%         i
%         FR_t_BO(:,:,i)
%         obj1selective = [obj1selective i];
%     elseif  (min(FR_t_BO(2,:,i))>minBound && max(max(FR_t_BO([1,3,4],:,i)))<maxBound)
%         i
%         FR_t_BO(:,:,i)
%         obj2selective = [obj2selective i];
%     elseif  (min(FR_t_BO(3,:,i))>minBound && max(max(FR_t_BO([1,2,4],:,i)))<maxBound)
%         i
%         FR_t_BO(:,:,i)
%         obj3selective = [obj3selective i];
%     elseif  (min(FR_t_BO(4,:,i))>minBound && max(max(FR_t_BO([1,2,3],:,i)))<maxBound)
%         i
%         FR_t_BO(:,:,i)
%         obj4selective = [obj4selective i];
%     end
% end
% obj1selective
% obj2selective
% obj3selective
% obj4selective


%3408




%%2 objects
% count_u = 0;
% for i = 1:size(FR_u,3)
%     if (min(FR_u(1,:,i))>minBound && max(FR_u(2,:,i))<maxBound)
%         i
%         FR_u(:,:,i)
%         count_u = count_u+1;
%     elseif (min(FR_u(2,:,i))>minBound && max(FR_u(1,:,i))<maxBound)
%         i
%         FR_u(:,:,i)
%         count_u = count_u+1;
%     end
% end
% 
% count_t = 0;
% for i = 1:size(FR_t,3)
%     if (min(FR_t(1,:,i))>minBound && max(FR_t(2,:,i))<maxBound)
%         i
%         FR_t(:,:,i)
%         count_t = count_t+1;
%         'test'
%     elseif  (min(FR_t(2,:,i))>minBound && max(FR_t(1,:,i))<maxBound)
%         i
%         FR_t(:,:,i)
%         count_t = count_t+1;
%         'test'
%     end
% end
%[count_u count_t]






% %% convert FR struct to combined to single array
% FR_u_conv = zeros(1,8,64*64);
% FR_u_conv(1,1:4,:) = FR_u(1,:,:);
% FR_u_conv(1,5:8,:) = FR_u(2,:,:);
% 
% FR_t_conv = zeros(1,8,64*64);
% FR_t_conv(1,1:4,:) = FR_t(1,:,:);
% FR_t_conv(1,5:8,:) = FR_t(2,:,:);
% 
% figure;
% subplot(2,1,1)
% plot(reshape(FR_u_conv(1,:,3408),1,8),'k','LineWidth',2)
% hold on
% plot(reshape(FR_u_conv(1,:,51),1,8),'--k','LineWidth',2)
% plot(reshape(FR_u_conv(1,:,1308),1,8),'.-k','LineWidth',2)
% plot(reshape(FR_u_conv(1,:,3554),1,8),':k','LineWidth',2)
% xlim([0.8 8.2])
% ylim([-5 105])
% ylabel('Untrained Network');
% 
% subplot(2,1,2)
% plot(reshape(FR_t_conv(1,:,3408),1,8),'k','LineWidth',2)
% hold on
% plot(reshape(FR_t_conv(1,:,51),1,8),'--k','LineWidth',2)
% plot(reshape(FR_t_conv(1,:,1308),1,8),'.-k','LineWidth',2)
% plot(reshape(FR_t_conv(1,:,3554),1,8),':k','LineWidth',2)
% xlim([0.8 8.2])
% ylim([-5 105])
% ylabel('Trained Network')
% xlabel(['cell: 3408, 51, 1308, 3554']);