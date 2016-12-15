function PGStat = polyAfterAnalysis(name)
    load(['../output/' name '/groups_trained.mat']);
%     load(['../output/' name '/groups_untrained.mat']);
    nPGs = length(groups);

    PGStat = zeros(2,nPGs);

    for i=1:nPGs
        PGStat(1,i) = length(groups{1,i}.firings);
        PGStat(2,i) = groups{1,i}.longest_path;
    end
end

figure;
tmp = [stat11t stat12t stat13t stat14t stat14u];
labels = [];
for i =1:length(stat11t) labels = [labels; 1]; end
for i =1:length(stat12t) labels = [labels; 2]; end
for i =1:length(stat13t) labels = [labels; 3]; end
for i =1:length(stat14t) labels = [labels; 4]; end
for i =1:length(stat14u) labels = [labels; 5]; end

subplot(1,2,1);
%boxplot(tmp(1,:), labels, 'Labels', {'FF','FF+FB','FF+LAT','FF+FB+LAT','FF+FB+LAT (ut)'}); h=findobj(gca,'tag','Outliers');delete(h);ylim([0 18]);title('Average Size of PGs');ylabel('Number of Spikes involved in PGs');
boxplot(tmp(1,:), labels); h=findobj(gca,'tag','Outliers');delete(h);ylim([0 18]);title('Average Size of PGs');ylabel('Number of Spikes involved in PGs');
hold on
plot([1 2 3 4 5],[mean(stat11t(1,:)) mean(stat12t(1,:)) mean(stat13t(1,:)) mean(stat14t(1,:)) mean(stat14u(1,:))],'o', 'color', 'r');
plot([4.5 4.5],[0 18],'k--');


subplot(1,2,2);
%boxplot(tmp(2,:), labels, 'Labels', {'FF','FF+FB','FF+LAT','FF+FB+LAT','FF+FB+LAT (ut)'}); h=findobj(gca,'tag','Outliers');delete(h);ylim([0 6]);title('Maximum Size of PGs');ylabel('Average Maximum Length of PGs');
boxplot(tmp(2,:), labels); h=findobj(gca,'tag','Outliers');delete(h);ylim([0 6]);title('Maximum Size of PGs');ylabel('Average Maximum Length of PGs');

hold on
plot([1 2 3 4 5],[mean(stat11t(2,:)) mean(stat12t(2,:)) mean(stat13t(2,:)) mean(stat14t(2,:)) mean(stat14u(2,:))],'o', 'color', 'r');
plot([4.5 4.5],[0 6],'k--');