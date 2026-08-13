clc; clear; close all;

% Get the list of files in "Images"
file_patterns = {'*.bmp', '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif'};

subfolder = '.\Images';
if ~exist(subfolder, 'dir')
    fprintf('Error: Subfolder "Images" is not found');
    return;
end

raw_files = cell(numel(file_patterns), 1);
for p = 1:numel(file_patterns)
    raw_files{p} = dir(fullfile(subfolder, file_patterns{p}));
end

files = vertcat(raw_files{:});
img_types = {'Gray', 'RGB'};

for k = 1:numel(files)
    filename = files(k).name;
    [~, name, ext] = fileparts(filename);
    ext_lower = lower(ext);
    
    if ~endsWith(name, '_outm')
        full_input_path = fullfile(subfolder, filename);
        full_output_path = fullfile(subfolder, [name, '_outm', ext]);
        
        try
            % 1. Read input image directly
            img = imread(full_input_path);
            
            % 2. Bit depth detection
            is_tiff = any(strcmpi(ext_lower, {'.tiff', '.tif'}));
            
            if is_tiff
                % Use lightning-fast native LibTIFF interface to query the exact tag
                t = Tiff(full_input_path, 'r');
                raw_depth = t.getTag('BitsPerSample');
                bit_depth = raw_depth(1); % Extract scalar if TIFF is RGB
                t.close();
            elseif isa(img, 'uint16')
                bit_depth = 16;
            else
                bit_depth = 8;
            end
            
            % 3. Normalization and Color Space handling (RGB to HSV)
            max_val = (2^bit_depth) - 1;
            is_color = (size(img, 3) == 3);
            
            if is_color
                % Normalize full RGB image to [0..1]
                img_rgb_double = double(img) / max_val;
                
                % Use native high-speed MATLAB conversion to HSV space
                img_hsv = rgb2hsv(img_rgb_double);
                
                % Extract Value (Brightness) channel for our filter
                img_normalized = img_hsv(:,:,3);
            else
                % Standard grayscale normalization
                img_normalized = double(img(:,:,1)) / max_val;
            end
            
            % 4. Process through ARC-filter (operating on normalized Grayscale 0..1)
            img_processed = ARC_filter(img_normalized);
            
            % Strict math clipping to prevent any mathematical overflow
            img_processed = max(0, min(1, img_processed));
            
            % 5. Reconstruction and saving
            if is_color
                % Reconstruct the HSV image with the processed Value channel
                img_hsv(:,:,3) = img_processed;
                
                % Convert back to native RGB double
                img_rgb_out = hsv2rgb(img_hsv);
                
                % Quantize back to original bit-depth
                if bit_depth > 8 && is_tiff
                    img_out = uint16(round(img_rgb_out * max_val));
                else
                    img_out = uint8(round(img_rgb_out * 255));
                end
            else
                % Standard grayscale quantization
                if is_tiff && bit_depth > 8
                    img_out = uint16(round(img_processed * max_val));
                else
                    img_out = uint8(round(img_processed * 255));
                end
            end
            
            % Write to disk
            if ~is_color && any(strcmpi(ext_lower, {'.jpg', '.jpeg'}))
                imwrite(img_out, full_output_path, 'ColorSpace', 'grayscale');
            else
                imwrite(img_out, full_output_path); % RGB arrays save automatically without flags
            end
            
            fprintf('Processed successfully: %s (%d-bit %s -> %s)\n', ...
                filename, bit_depth, img_types{is_color + 1}, [name, '_outm', ext]);
            
        catch ME
            fprintf('Error processing %s: %s\n', filename, ME.message);
        end
    end
end

fprintf('\nAll images processed!\n');

%% =====================================================================
% Adaptive Rank-Cluster (ARC-)filter
% ======================================================================
function out = ARC_filter(img)

    eps_safe = 1e-16;
    alpha = 1.0 ./ sqrt(2);
    beta = 1.0 ./ sqrt(2*pi);
    [rows, cols] = size(img);
    num_windows = rows * cols;
       
    % 1. Fast vectorization of 3x3 sliding window
    B = im2col(padarray(img, [1 1], 'symmetric'), [3 3], 'sliding');
    center_vals = B(5, :); 
    B = sort(B, 1);        
    total_mu = B(5, :);
    
    % Truncate the sorted set to 7 elements (rows 2..8)
    B_trunc = B(2:8, :);
    clear B; 
    
    % 2. Otsu thresholding across 6 possible cuts
    cumsum_F = cumsum(B_trunc, 1);
    sum_total_7 = cumsum_F(7, :);
    buf_A = cumsum_F(1:6, :);                  
    clear cumsum_F;
    
    % Matrix-free Otsu: optimized broadcasting inside CPU cache
    MV = [1/3; 1/5; 1/6; 1/6; 1/5; 1/3];
    V = MV .* (buf_A .* 7.0 - (1:6).' .* sum_total_7).^2;
    [~, N1] = max(V, [], 1);
    N2 = 7.0 - N1;
    clear V;
    
    % Safe high-speed linear indexing
    stride = uint32(size(buf_A, 1));
    lin_opt = uint32(0:num_windows-1) * stride + uint32(N1);
    final_mu1 = buf_A(lin_opt) ./ N1;
    final_mu2 = (sum_total_7 - buf_A(lin_opt)) ./ N2;
    clear buf_A;

    % 3. Standard deviation calculation via single sum of squares
    cumsum_Sq = cumsum(B_trunc.^2, 1);
    clear B_trunc;
    
    buf_SqA = cumsum_Sq(1:6, :);
    sum_sq1_opt = buf_SqA(lin_opt);
    clear buf_SqA;
    
    sum_sq2_opt = cumsum_Sq(7, :) - sum_sq1_opt;
    clear cumsum_Sq lin_opt;
    
    sigma1 = max(eps_safe, sqrt(abs(N1 .* sum_sq1_opt - (N1 .* final_mu1).^2)) ./ N1);
    sigma2 = max(eps_safe, sqrt(abs(N2 .* sum_sq2_opt - (N2 .* final_mu2).^2)) ./ N2);
    clear sum_sq1_opt sum_sq2_opt N1 N2;

    % 4. Separation threshold (Sth) computation and application of the transformation function
    inv_sigma_dist = 1.0 ./ (sigma1 + sigma2);
    mu_dist = max(1.0, (final_mu2 - final_mu1).* inv_sigma_dist);
    Sth = (final_mu1 .* sigma2 + final_mu2 .* sigma1) .* inv_sigma_dist;
    clear final_mu1 final_mu2 inv_sigma_dist;
    
    msk = (center_vals < Sth);
    sigma_tot = sigma1 .* msk + sigma2 .* (~msk);
    clear sigma1 sigma2 msk;
    
    res = Sth + sigma_tot .* mu_dist .* erf(beta .* (center_vals - Sth) ./ sigma_tot);
    clear sigma_tot center_vals Sth;

    % 5. Soft cluster mixing and reshaping back to 2D matrix layout
    out = reshape(total_mu + erf(abs(alpha * (mu_dist - 1.0))) .* (res - total_mu), [rows, cols]);
    clear mu_dist total_mu res;
end