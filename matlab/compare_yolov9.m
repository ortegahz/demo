clear; close all;

c = 7; wh = 6300; wha = 6304;

out_q = load('/home/manu/tmp/output.txt');
out_ref = load('/home/manu/tmp/output0.txt');

% out_q = out_q(1:length(out_ref));
% out_ref = out_ref(1:length(out_q));

out_q = reshape(out_q, [wha, c])';
out_q = out_q(:, 1:wh)'; 
out_q = reshape(out_q, [], 1);

% error = sum((out_ref - out_q) .^ 2) / length(out_ref);

% Calculate cosine similarity
dot_product = dot(out_ref, out_q);
norm_ref = norm(out_ref);
norm_q = norm(out_q);
cosine_similarity = dot_product / (norm_ref * norm_q);

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(num2str(cosine_similarity));

subplot(3,1,2); plot(1:length(out_q), out_q, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

