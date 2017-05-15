experimentName = '6.3grayAndBlack_a--StatDec_FFLATFB_20EP_FB5_rand123';
layer = 1;
FRThreshold = 200.0;
nBins = 5;

%%load untrained FR
trainedNet = 0;
FR_u = loadFR(experimentName,layer,trainedNet);
FR_u(FR_u>FRThreshold) = FRThreshold;

trainedNet = 1;
FR_t = loadFR(experimentName,layer,trainedNet);
FR_t(FR_t>FRThreshold) = FRThreshold;

%% running InfoAnalysis
% IRs_u = infoAnalysis(FR_u,0,nBins);
% IRs_t = infoAnalysis(FR_t,1,nBins);

% %% convert FR struct to BO
FR_u_BO = zeros(4,2,64*64);
FR_u_BO(1,:,:) = FR_u(1,1:2,:);
FR_u_BO(2,:,:) = FR_u(1,3:4,:);
FR_u_BO(3,:,:) = FR_u(2,1:2,:);
FR_u_BO(4,:,:) = FR_u(2,3:4,:);

FR_t_BO = zeros(4,2,64*64);
FR_t_BO(1,:,:) = FR_t(1,1:2,:);
FR_t_BO(2,:,:) = FR_t(1,3:4,:);
FR_t_BO(3,:,:) = FR_t(2,1:2,:);
FR_t_BO(4,:,:) = FR_t(2,3:4,:);

IRs_u = infoAnalysis(FR_u_BO,0,nBins);
IRs_t = infoAnalysis(FR_t_BO,1,nBins);

%% Plot BO info
% hold off;
% plot(-1*sort(-1*max(transpose(IRs_t))),'k','LineWidth',2);
% hold on;
% plot(-1*sort(-1*max(transpose(IRs_u))),'--k','LineWidth',2);
% nCellsWithNonZeroInfo = length(find(IRs_t>0));
% axis([1 nCellsWithNonZeroInfo*1.2 -0.1 log2(4)+0.1]);
% title('Single Cell Information Analysis');
% ylabel('Information [bit]');
% xlabel('Cell Rank');

%% Plot BO info two objects at Loc1 combined
hold off;
IRs_u_loc1 = IRs_u(:,1:2);
plot(-0.5*sort(-1*max(transpose(IRs_u_loc1))),':k','LineWidth',2);
hold on;

% hold off;
% IRs_t_loc1 = IRs_t(:,1:2);
% plot(-0.5*sort(-1*max(transpose(IRs_t_loc1))),'--k','LineWidth',2);
% hold on;

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




% %% Find V4 Cells
% minBound = FRThreshold*0.5;
% maxBound = FRThreshold*0.5;
% count_obj1 = 0;
% count_obj2 = 0;
% for i = 1:size(FR_t,3)
%     if (min(FR_t(1,:,i))>minBound && max(FR_t(2,:,i))<maxBound)
%         i
%         FR_t(:,:,i)
%         count_obj1 = count_obj1+1;
%         'test'
%     elseif  (min(FR_t(2,:,i))>minBound && max(FR_t(1,:,i))<maxBound)
%         i
%         FR_t(:,:,i)
%         count_obj2 = count_obj2+1;
%         'test'
%     end
% end
% [count_obj1 count_obj2]






%% convert FR struct to combined to single array
FR_u_conv = zeros(1,8,64*64);
FR_u_conv(1,1:4,:) = FR_u(1,:,:);
FR_u_conv(1,5:8,:) = FR_u(2,:,:);

FR_t_conv = zeros(1,8,64*64);
FR_t_conv(1,1:4,:) = FR_t(1,:,:);
FR_t_conv(1,5:8,:) = FR_t(2,:,:);

%% PLOT BO FR
figure(1);
subplot(4,1,3)
hold off
plot(reshape(FR_u_conv(1,:,3408),1,8),'k','LineWidth',2)
hold on
plot(reshape(FR_u_conv(1,:,51),1,8),'--k','LineWidth',2)
plot(reshape(FR_u_conv(1,:,1308),1,8),'-.k','LineWidth',2)
plot(reshape(FR_u_conv(1,:,3554),1,8),':k','LineWidth',2)
xlim([0.8 8.2])
ylim([-5 55])
ylabel({'Untrained','Network'});

subplot(4,1,4)
hold off
plot(reshape(FR_t_conv(1,:,3408),1,8),'k','LineWidth',2)
hold on
plot(reshape(FR_t_conv(1,:,51),1,8),'--k','LineWidth',2)
plot(reshape(FR_t_conv(1,:,1308),1,8),'-.k','LineWidth',2)
plot(reshape(FR_t_conv(1,:,3554),1,8),':k','LineWidth',2)
xlim([0.8 8.2])
ylim([-5 55])
ylabel({'Trained','Network'})
xlabel(['cell: 3408, 51, 1308, 3554']);

% %% PLOT V4 FR
% figure(1);
% subplot(4,1,3)
% plot(reshape(FR_u_conv(1,:,192),1,8),'k','LineWidth',2)
% hold on
% plot(reshape(FR_u_conv(1,:,305),1,8),'--k','LineWidth',2)
% xlim([0.8 8.2])
% ylim([-5 55])
% ylabel({'Untrained','Network'});
% 
% 
% subplot(4,1,4)
% plot(reshape(FR_t_conv(1,:,192),1,8),'k','LineWidth',2)
% hold on
% plot(reshape(FR_t_conv(1,:,305),1,8),'--k','LineWidth',2)
% xlim([0.8 8.2])
% ylim([-5 55])
% ylabel({'Trained','Network'})
% xlabel(['cell: 15552, 15665']);