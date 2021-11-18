%% plot svm data here
figure(); 
plot(floor(props*200), 100*svm_err_PCA(1,:), 'bo-', 'DisplayName', 'RBF');
hold on; 
plot(floor(props*200), 100*svm_err_PCA(2,:), 'ro-', 'DisplayName', 'Polynomial');
hold on; 
plot(floor(props*200), 100*svm_err_PCA(3,:), 'go-', 'DisplayName', 'Linear');
hold on; 
plot(floor(props*200), 100*svm_err_PCA(5,:), 'ko-', 'DisplayName', '5 times Boosted Linear');
xlabel("Number of training points per class", 'Interpreter', 'latex'); 
ylabel("Percentage testing error", 'Interpreter', 'latex'); 
set(gca, 'FontSize', 16);
ylim([10, 60]);
