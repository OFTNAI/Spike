function PGStat = polyAfterAnalysis(name)
    load(['../output/' name '/groups_trained.mat']);
    nPGs = length(groups);

    PGStat = zeros(2,nPGs);

    for i=1:nPGs
        PGStat(1,i) = length(groups{1,i}.firings);
        PGStat(2,i) = groups{1,i}.longest_path;
    end
end


%tmp = [stat11 stat12 stat13 stat14];
% labels = [];
% for i =1:length(stat11) labels = [labels; 1]; end
% for i =1:length(stat12) labels = [labels; 2]; end
% for i =1:length(stat13) labels = [labels; 3]; end
% for i =1:length(stat14) labels = [labels; 4]; end

%subplot(1,2,1);
%boxplot(tmp(1,:), labels, 'Labels', {'FF','FF+FB','FF+LAT','FF+FB+LAT'}); h=findobj(gca,'tag','Outliers');delete(h);ylim([0 18]);title('Average Size of PGs');ylabel('Number of Spikes involved in PGs');
%hold on
%plot([1 2 3 4],[mean(stat11(1,:)) mean(stat12(1,:)) mean(stat13(1,:)) mean(stat14(1,:))],'o', 'color', 'r');

%subplot(1,2,2);
% boxplot(tmp(2,:), labels, 'Labels', {'FF','FF+FB','FF+LAT','FF+FB+LAT'}); h=findobj(gca,'tag','Outliers');delete(h);ylim([0 6]);title('Maximum Size of PGs');ylabel('Average Maximum Length of PGs');
% hold on
% plot([1 2 3 4],[mean(stat11(2,:)) mean(stat12(2,:)) mean(stat13(2,:)) mean(stat14(2,:))],'o', 'color', 'r');
