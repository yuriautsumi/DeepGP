function [m_adapt, s_adapt] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_s, s_s, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha)
% adaptation 
m_adapt = []; % to append, do a = [a; num_to_append] 
s_adapt = []; 

% convert hyperparameters to MATLAB format 
hyp_source.lik = sn2; % testing 
hyp_source.cov(1) = ls; 
hyp_source.cov(2) = var; 
% hyp_source.lik = 0.5*log(sn2); 
% hyp_source.cov(1) = log(ls); 
% hyp_source.cov(2) = 0.5*log(var); 
slices = mul; % TODO: incorporate slices 

%%% BEGIN ADAPTATION %%%

a = size(x_a); 

for i = 0:a(1,1) %1 to 20 
% for i = 1:2 %1 to 20 
    if i == 0
        m_adapt = [m_adapt; m_s(1)]; 
        s_adapt = [s_adapt; s_s(1)]; 
        continue 
    end
    
    y_a_patient = y_a(1:i);
    
    % Adaptation calculations 
    % K_ts, K_tt 
    K_ts = K_ts_star_all(1:end, 1:i);
    K_tt = K_tt_all(1:i, 1:i); 
    
    % alpha_adapt 
    V = L\K_ts; % or try L'\K_ts'; 
    mu_t = K_ts'*alpha; 
    K_tt_dim = size(K_tt); 
    K_tt_dim = K_tt_dim(1);
    C_t = K_tt - V'*V + sn2*eye(K_tt_dim); % checked, ok 
    L_adapt = jitterChol(C_t); 
    alpha_adapt = solve_chol(L_adapt,(y_a_patient - mu_t));
    
    % V_adapt 
    K_t_star = K_t_star_all(1:i, i+1:i+1);
    K_ts_star = K_ts_star_all(1:end, i+1:i+1);
    V_star = L\K_ts_star; % or try L'\K_ts_star' 
    C_t_star = K_t_star - V'*V_star; 
    V_adapt = L_adapt'\C_t_star; % or try L_adapt'\C_t_star' 
    
%     disp('L')
%     disp(L)
%     
%     disp('alpha')
%     disp(alpha)
%     
%     disp('K_ts')
%     disp(K_ts)
%     
%     disp('mu_t')
%     disp(mu_t)
%     
%     disp('C_t')
%     disp(C_t)
%     
%     disp('L_adapt')
%     disp(L_adapt)
%     
%     disp('(y_a_patient - mu_t)')
%     disp((y_a_patient - mu_t))
%     
%     disp('alpha_adapt')
%     disp(alpha_adapt)
%     
%     disp('add mu_adapt')
%     disp(C_t_star'*alpha_adapt)
%     
%     disp('add sigma_adapt')
%     disp(sum(V_adapt.*V_adapt)')
    
    m_adapt_ele = m_s(i+1,1) + C_t_star'*alpha_adapt;
    s_adapt_ele = s_s(i+1,1) - sum(V_adapt.*V_adapt)'; 
    
    m_adapt = [m_adapt; m_adapt_ele]; 
    s_adapt = [s_adapt; s_adapt_ele];
end 
end 