clear;
clc;
close all;

station = 'IBRH13';
net = 'CNN';
motiontype = 'simulate';
filedir = fullfile([station, '_results'], [motiontype, '_', net]);
data = load(fullfile(filedir, 'result.mat'));

motionfile = fopen(fullfile([station, '_results'], 'selectevent.txt'));
eventlist = [];
while ~feof(motionfile)
    eventlist = [eventlist; fgetl(motionfile)];
end

% Loss
train_loss = data.train_loss;
valid_loss = data.valid_loss;
iters = 1 : size(train_loss, 2);
figure();
hold on;
plot(iters, train_loss, 'linewidth', 2);
plot(iters, valid_loss, 'linewidth', 2);
hold off;
xlabel('Iterations');
ylabel('Loss');
legend('Train', 'Valid');
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
% savefig(fullfile(filedir, 'loss.fig'));
% saveas(gcf, fullfile(filedir, 'loss.emf'));

Period = logspace(-2, 1, 101);
t = data.time;
dt = data.dt;

% For train data
dhacc = data.train_data;
upacc = data.train_label;
pracc = data.train_pred_tt;

r_train = zeros(size(dhacc, 1), 1);
err_train = zeros(size(dhacc, 1), 1);
for i = 1 : size(dhacc, 1)
    r = corrcoef(upacc(i, :), pracc(i, :));
    r_train(i) = r(2);
    err_train(i) = mean(((upacc(i, :) - pracc(i, :)) / max(abs(upacc(i, :)))).^2);
end
figure();
histogram(r_train, [0 : 0.05 : 1]);
xlabel('Correlation Coefficient');
ylabel('Counts');
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
% savefig(fullfile(filedir, 'train_r.fig'));
% saveas(gcf, fullfile(filedir, 'train_r.emf'));

figure();
histogram(err_train, [0 : 0.0015 : 0.03]);
xlabel('RMSE');
ylabel('Counts');
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
% savefig(fullfile(filedir, 'train_err.fig'));
% saveas(gcf, fullfile(filedir, 'train_err.emf'));

train_loss = fopen(fullfile(filedir, 'train_loss.out'), 'w');
for i = 1 : size(dhacc, 1)
    loss = mean(((upacc(i, :) - pracc(i, :)) / max(abs(dhacc(i, :)))).^2);
    pga = max(abs(upacc(i, :))) / 981;
    fprintf(train_loss, '%s\t%.6f\t%.6f\t%.6f\t%.4f\n', eventlist(data.train_idx(i)+1, :), loss, r_train(i), err_train(i), pga);
end


% for i = 1 : size(dhacc, 1)
%     dhSa = getResponseSpectrum(dhacc(i, :)', dt);
%     upSa = getResponseSpectrum(upacc(i, :)', dt);
%     prSa = getResponseSpectrum(pracc(i, :)', dt);
%     [f, dhfouri] = PlotFourierSpectrum(dhacc(i, :)', dt);
%     [f, upfouri] = PlotFourierSpectrum(upacc(i, :)', dt);
%     [f, prfouri] = PlotFourierSpectrum(pracc(i, :)', dt);
%     upTF = upfouri ./ dhfouri;
%     prTF = prfouri ./ dhfouri;
%     pga = max(abs(upacc(i, :))) / 981;
% 
%     figure();
%     subplot(4, 2, [1,2]);
%     plot(t, dhacc(i, :), 'k');
%     ylabel('acc (gal)');
%     xlim([0, 60]);
%     legend('Downhole');
%     set(gca, 'fontsize', 16, 'fontname', 'Times New Roman');
%     subplot(4, 2, [3, 4]);
%     plot(t, upacc(i, :), 'b', t, pracc(i, :), 'r');
%     xlabel('t (s)');
%     ylabel('acc (gal)');
%     xlim([0, 60]);
%     legend('Recorded', 'Predicted');
%     set(gca, 'fontsize', 16, 'fontname', 'Times New Roman');
%     subplot(4, 2, [5,7]);
%     loglog(f, upTF, 'b', f, prTF, 'r');
%     xlabel('Freq (Hz)');
%     ylabel('Transfer Function');
%     xlim([0.5, 15]);
%     xticks([0.01, 0.1, 1, 10]);
%     legend('Recorded', 'Predicted');
%     set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
%     subplot(4, 2, [6,8]);
%     semilogx(Period, upSa, 'b', Period, prSa, 'r');
%     xlabel('T (s)');
%     ylabel('Sa (gal)');
%     xlim([0.01, 20]);
%     xticks([0.01, 0.1, 1, 10]);
%     legend('Recorded', 'Predicted');
%     set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
%     set(gcf, 'Position', [283,226,1000,750]);
%     % savefig(fullfile(filedir, 'figures', ['train', num2str(i), '_r', sprintf('%.3f', r_train(i)), '_e', sprintf('%.3f', err_train(i)), '_PGA', sprintf('%.3f', pga), '_', eventlist(data.train_idx(i)+1, :), '.fig']));
%     % saveas(gcf, fullfile(filedir, 'figures', ['train', num2str(i), '_r', sprintf('%.3f', r_train(i)), '_e', sprintf('%.3f', err_train(i)), '_PGA', sprintf('%.3f', pga), '_', eventlist(data.train_idx(i)+1, :), '.emf']));
%     close(figure(gcf));
% end


% For test data
dhacc = data.test_data;
upacc = data.test_label;
pracc = data.test_pred_tt;

r_test = zeros(size(dhacc, 1), 1);
err_test = zeros(size(dhacc, 1), 1);
for i = 1 : size(dhacc, 1)
    r = corrcoef(upacc(i, :), pracc(i, :));
    r_test(i) = r(2);
    err_test(i) = mean(((upacc(i, :) - pracc(i, :)) / max(abs(upacc(i, :)))).^2);
end
figure();
histogram(r_test, [0 : 0.05 : 1], 'FaceColor', '#D95319');
xlabel('Correlation Coefficient');
ylabel('Counts');
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
% savefig(fullfile(filedir, 'test_r.fig'));
% saveas(gcf, fullfile(filedir, 'test_r.emf'));

figure();
histogram(err_test, [0 : 0.0015 : 0.03], 'FaceColor', '#D95319');
xlabel('RMSE');
ylabel('Counts');
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
% savefig(fullfile(filedir, 'test_err.fig'));
% saveas(gcf, fullfile(filedir, 'test_err.emf'));

test_loss = fopen(fullfile(filedir, 'test_loss.out'), 'w');
for i = 1 : size(dhacc, 1)
    loss = mean(((upacc(i, :) - pracc(i, :)) / max(abs(dhacc(i, :)))).^2);
    pga = max(abs(upacc(i, :))) / 981;
    fprintf(test_loss, '%s\t%.6f\t%.6f\t%.6f\t%.4f\n', eventlist(data.train_idx(i)+1, :), loss, r_test(i), err_test(i), pga);
end

% for i = 1 : size(dhacc, 1)
%     dhSa = getResponseSpectrum(dhacc(i, :)', dt);
%     upSa = getResponseSpectrum(upacc(i, :)', dt);
%     prSa = getResponseSpectrum(pracc(i, :)', dt);
%     [f, dhfouri] = PlotFourierSpectrum(dhacc(i, :)', dt);
%     [f, upfouri] = PlotFourierSpectrum(upacc(i, :)', dt);
%     [f, prfouri] = PlotFourierSpectrum(pracc(i, :)', dt);
%     upTF = upfouri ./ dhfouri;
%     prTF = prfouri ./ dhfouri;
%     pga = max(abs(upacc(i, :))) / 981;
% 
%     figure();
%     subplot(4, 2, [1,2]);
%     plot(t, dhacc(i, :), 'k');
%     ylabel('acc (gal)');
%     xlim([0, 60]);
%     legend('Downhole');
%     set(gca, 'fontsize', 16, 'fontname', 'Times New Roman');
%     subplot(4, 2, [3, 4]);
%     plot(t, upacc(i, :), 'b', t, pracc(i, :), 'r');
%     xlabel('t (s)');
%     ylabel('acc (gal)');
%     xlim([0, 60]);
%     legend('Recorded', 'Predicted');
%     set(gca, 'fontsize', 16, 'fontname', 'Times New Roman');
%     subplot(4, 2, [5,7]);
%     loglog(f, upTF, 'b', f, prTF, 'r');
%     xlabel('Freq (Hz)');
%     ylabel('Transfer Function');
%     xlim([0.5, 15]);
%     xticks([0.01, 0.1, 1, 10]);
%     legend('Recorded', 'Predicted');
%     set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
%     subplot(4, 2, [6,8]);
%     semilogx(Period, upSa, 'b', Period, prSa, 'r');
%     xlabel('T (s)');
%     ylabel('Sa (gal)');
%     xlim([0.01, 20]);
%     xticks([0.01, 0.1, 1, 10]);
%     legend('Recorded', 'Predicted');
%     set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
%     set(gcf, 'Position', [283,226,1000,750]);
%     % savefig(fullfile(filedir, 'figures', ['test', num2str(i), '_r', sprintf('%.3f', r_test(i)), '_e', sprintf('%.3f', err_test(i)), '_PGA', sprintf('%.3f', pga), '_', eventlist(data.train_idx(i)+1, :), '.fig']));
%     % saveas(gcf, fullfile(filedir, 'figures', ['test', num2str(i), '_r', sprintf('%.3f', r_test(i)), '_e', sprintf('%.3f', err_test(i)), '_PGA', sprintf('%.3f', pga), '_', eventlist(data.train_idx(i)+1, :), '.emf']));
%     close(figure(gcf));
% end