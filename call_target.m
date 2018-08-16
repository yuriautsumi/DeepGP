function [m_target, s_target] = call_target(x_a, y_a, x_s, y_s, xtest, m_s, s_s, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all)
% note: K_t_star_all is K_ts_star_all. K_tt_all is K_s_all. 
% target  
m_target = []; % to append, do a = [a; num_to_append] 
s_target = []; 

% convert hyperparameters to MATLAB format 
% hyp_source.lik = sn2; % testing 
% hyp_source.cov(1) = ls; 
% hyp_source.cov(2) = var; 
hyp_source.lik = 0.5*log(sn2); 
hyp_source.cov(1) = log(ls); 
hyp_source.cov(2) = 0.5*log(var); 
slices = mul; % TODO: incorporate slices 

%%% BEGIN TARGET %%%

a = size(x_a); 

for i = 0:a(1,1) %0 to 20 
% for i = 0:2 %0 to 20 
    if i == 0
        m_target = [m_target; m_s(1)]; 
        s_target = [s_target; s_s(1)]; 
        continue 
    end
    
    y_a_patient = y_a(1:i);
    
    % Target calculations 
    % K_ts_star 
    K_ts_star = K_t_star_all(1:i, i+1:i+1);
    
    % k_star_star 
    k_star_star = k_star_star_all(1,i);
    
    % V_star 
    K_s = K_tt_all(1:i, 1:i);
    K_s_dim = size(K_s);
    K_s_dim = K_s_dim(1);
    L = jitterChol(K_s + sn2*eye(K_s_dim));
    V_star = L'\K_ts_star;
    
    % alpha 
    alpha_denom = L'\y_a_patient;
    alpha = L\alpha_denom; 
    
    m_target_ele = K_ts_star'*alpha;
    s_target_ele = k_star_star - sum(V_star.*V_star)';
    
    m_target = [m_target; m_target_ele];
    s_target = [s_target; s_target_ele]; 
end 

end 