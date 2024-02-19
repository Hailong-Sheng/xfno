function value = rbf_intp(x, node, coef)

[x_size,dim] = size(x);
[n_size,~] = size(node);

ee = 1e-2;

value = zeros(x_size,1);
for i = 1:n_size
    dist = (sum((x-node(i,:)).^2,2)).^ 0.5;
    phi = (ee + dist.^2).^ (-0.5);
    value = value + coef(i) * phi;
end
for i = 1:dim
    value = value + coef(n_size+i) * x(:,i);
end
value = value + coef(end);