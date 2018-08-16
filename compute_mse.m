function mse = compute_mse(g_t, y_pred)

dim = size(g_t);
dim = dim(1); 
mse = sum(abs(y_pred-g_t))/dim; 

end 