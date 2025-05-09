
input_csv = 'trimmed_prot_no_head.csv';       % input file
output_csv = 'corrected_spectra.csv';         % output file
fig_dir = 'baseline_figs';                    % folder for .fig files

lambda = 1e5;         % Smoothness penalty
p = 0.01;             % Asymmetry
niter = 10;           % Number of AsLS iterations
% -------------------------------------------------------

% Create figure directory if needed
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

% Load data
data = readmatrix(input_csv);
mz = data(:, 1);              % m/z = column 1
spectra = data(:, 2:end);     % other columns = intensities

% initialize corrected spectra
[n_points, n_samples] = size(spectra);
corrected = zeros(n_points, n_samples);

% baseline correction and save .fig plots
for i = 1:n_samples
    y = spectra(:, i);
    baseline = als_baseline(y, lambda, p, niter);
    corrected(:, i) = y - baseline;

    % plot and save figure
    fig = figure('Visible', 'on');
    plot(mz, y, 'k'); hold on;
    plot(mz, baseline, 'r--');
    plot(mz, corrected(:, i), 'b');
    xlabel('m/z'); ylabel('Intensity');
    title(['Sample ', num2str(i)]);
    legend('Original', 'Baseline', 'Corrected');
    axis tight;
    drawnow;

    savefig(fig, fullfile(fig_dir, sprintf('sample_%03d.fig', i)));
    close(fig);
end

% save data to file
writematrix([mz corrected], output_csv);



% ----------------- Baseline Function -------------------
function baseline = als_baseline(y, lambda, p, niter)
    L = length(y);
    D = diff(speye(L), 2);
    H = lambda * (D' * D);
    w = ones(L, 1);
    for i = 1:niter
        W = spdiags(w, 0, L, L);
        Z = W + H;
        baseline = Z \ (w .* y);
        w = p * (y > baseline) + (1 - p) * (y < baseline);
    end
end
