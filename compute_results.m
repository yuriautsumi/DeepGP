function compute_results(model_num) 
% guideline function for creating compute results functions 

% get current directory 
currentFolder = pwd; 

% get list of test patient IDs for each fold 
ids_original = csvread([currentFolder '/Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv']);
ids = reshape(ids_original, 10, 10); 

% initialize arrays 
ID_all = [];

g_t = []; 

m_source = [];
m_adapt = [];
m_target = []; 

s_adapt = [];
s_target = []; 

error = []; 

% for each fold: 
% for fold = 1:2 % 1 to 10 folds 
for fold = 1:10 % 1 to 10 folds 
    % find folder 
    foldFolder = [currentFolder '/m' num2str(model_num) '_mat/fold_' num2str(fold)]; 
    
    % get patients 
    fold_ids = ids(1:end, fold);
    
    % for each test patient in the fold: 
    for pat = fold_ids'
        % load patient variables 
        x_a = load([foldFolder '/id_' num2str(pat) '_x_a.mat']); 
        y_a = load([foldFolder '/id_' num2str(pat) '_y_a.mat']); 
        x_s = load([foldFolder '/id_' num2str(pat) '_x_s.mat']); 
        y_s = load([foldFolder '/id_' num2str(pat) '_y_s.mat']); 
        xtest = load([foldFolder '/id_' num2str(pat) '_xtest.mat']); 
        m_s = load([foldFolder '/id_' num2str(pat) '_m_s.mat']); 
        s_s = load([foldFolder '/id_' num2str(pat) '_s_s.mat']); 
        
        ls = load([foldFolder '/id_' num2str(pat) '_ls.mat']);
        mul = load([foldFolder '/id_' num2str(pat) '_mul.mat']);
        var = load([foldFolder '/id_' num2str(pat) '_var.mat']);
        sn2 = load([foldFolder '/id_' num2str(pat) '_sn2.mat']);
        
        K_ts_star_all = load([foldFolder '/id_' num2str(pat) '_K_ts_star_all.mat']);
        K_tt_all = load([foldFolder '/id_' num2str(pat) '_K_tt_all.mat']);
        K_t_star_all = load([foldFolder '/id_' num2str(pat) '_K_t_star_all.mat']);
        L = load([foldFolder '/id_' num2str(pat) '_L.mat']);
        alpha = load([foldFolder '/id_' num2str(pat) '_alpha.mat']);
        
        k_star_star_all = load([foldFolder '/id_' num2str(pat) '_k_star_star_all.mat']);
        
        g_t_patient = load([foldFolder '/id_' num2str(pat) '_g_t_patient.mat']);
        
        x_a = x_a.x_a; 
        y_a = y_a.y_a; 
        x_s = x_s.x_s; 
        y_s = y_s.y_s; 
        xtest = xtest.xtest; 
        m_s = m_s.m_s; 
        s_s = s_s.s_s; 
        
        ls = ls.ls;
        mul = mul.mul;
        var = var.var;
        sn2 = sn2.sn2;
        
        K_ts_star_all = K_ts_star_all.K_ts_star_all;
        K_tt_all = K_tt_all.K_tt_all;
        K_t_star_all = K_t_star_all.K_t_star_all;
        L = L.L;
        alpha = alpha.alpha;
        
        k_star_star_all = k_star_star_all.k_star_star_all; 
        
        g_t_patient = g_t_patient.g_t_patient; 
        
        % get mean and variance predictions 
        [m_adapt_pat, s_adapt_pat] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_s, s_s, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha);
        [m_target_pat, s_target_pat] = call_target(x_a, y_a, x_s, y_s, xtest, m_s, s_s, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all);
        
        % get error 
        pat_ind = find(fold_ids' == pat); 
        m_source_pat = m_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
        source_m_dim = size(m_source_pat); 
        source_m_dim = source_m_dim(1); 
        source_pat_error = sum(abs(m_source_pat-g_t_patient))/source_m_dim; 
        
        adapt_m_dim = size(m_adapt_pat);
        adapt_m_dim = adapt_m_dim(1); 
        adapt_pat_error = sum(abs(m_adapt_pat-g_t_patient))/adapt_m_dim; 

        target_m_dim = size(m_target_pat);
        target_m_dim = target_m_dim(1); 
        target_pat_error = sum(abs(m_target_pat-g_t_patient))/target_m_dim; 
        
        % assesmble arrays 
        ID_all = [ID_all; ones(source_m_dim,1)*pat]; 
        g_t = [g_t; g_t_patient];
        m_adapt = [m_adapt; m_adapt_pat]; 
        m_target = [m_target; m_target_pat]; 
        s_adapt = [s_adapt; s_adapt_pat];
        s_target = [s_target; s_target_pat]; 
        error = [error; source_pat_error adapt_pat_error target_pat_error];
        
    end 
    
    % assemble arrays 
    m_source = [m_source; m_s]; 
    
end  

% assemble arrays 
results = [ID_all g_t m_source m_adapt m_target]; 
errors = [ids_original(1:20,1:end) error]; 

% write results to CSV 
csvwrite([currentFolder '/model_' num2str(model_num) '/results.csv'], results);
csvwrite([currentFolder '/model_' num2str(model_num) '/errors.csv'], errors);
end 