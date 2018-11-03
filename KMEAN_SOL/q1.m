X = csvread('HW2_Q2_X.txt');
y = csvread('HW2_Q2_y.txt');
idx = 41:640;
idx2 = 1:40;
[U S V] = svd(X, 'econ');

ridx = 1 : 20 : 601;
r = zeros(size(ridx));
for j = 1 : length(ridx)
i = ridx(j);
r(j) = -(U(idx2, 1:i)'*y(idx2))'*(U(idx, 1:i)'*y(idx));
end

plot(ridx, r);
xlabel('k');
ylabel('r(k)');