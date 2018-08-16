function compute_final_results(model_num, start_fold, end_fold) 
% use for models 0, 2 

disp('Computing final results...')

% get current directory 
currentFolder = pwd; 

% get list of test patient IDs for each fold 
ids_original = csvread([currentFolder '/Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv']);
ids = reshape(ids_original, 10, 10); 

% initialize arrays 
ID_all = [];

g_t = []; 

if model_num == 2
    m_base = [];
end 

m_source = [];
m_adapt = [];
m_target = []; 

s_source = [];
s_adapt = [];
s_target = []; 

tst_inds = []; 

error = []; 

% for each fold: 
% for fold = 1:2 % 1 to 10 folds 
for fold = start_fold:end_fold % 1 to 10 folds 
    
    disp(['Fold ' num2str(fold) ' in progress...']) 
    
    % find folder 
    foldFolder = [currentFolder '/kgp_results/m' num2str(model_num) '_mat/fold_' num2str(fold) '/final']; 
    
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
        
        g_t_patient = load([foldFolder '/id_' num2str(pat) '_g_t_patient.mat']);
        g_t_patient = g_t_patient.g_t_patient; 
        
        tst_ind_patient = load([foldFolder '/id_' num2str(pat) '_tst_ind_patient.mat']);
        tst_ind_patient = tst_ind_patient.tst_ind_patient; 
        
        if model_num == 2
            m_b_patient = load([foldFolder '/id_' num2str(pat) '_m_b_patient.mat']); 
            m_b_patient = m_b_patient.m_b_patient;
            base_pat_error = compute_mse(g_t_patient, m_b_patient); 
            m_base = [m_base; m_b_patient];
        end 
        
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
        
        x_a = x_a.x_a; 
        y_a = y_a.y_a; 
        x_s = x_s.x_s; 
        y_s = y_s.y_s; 
        xtest = xtest.xtest; 
        m_s = m_s.m_s; 
        s_s = s_s.s_s; 
        
        b = size(m_s);
        
        if b(1) < b(2)
            m_s = m_s';
            s_s = s_s';
        end 
        
        % get m_s / s_s data corresponding to patient 
        pat_ind = find(fold_ids' == pat); 
        m_source_pat = m_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
        s_source_pat = s_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
        
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
        
        % get mean and variance predictions 
        [m_adapt_pat, s_adapt_pat] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha);
        [m_target_pat, s_target_pat] = call_target(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all);
        
        % get error 
        non_zero_inds = find(tst_ind_patient~=0);
        g_t_pat_processed = g_t_patient(non_zero_inds); 
        m_source_pat_processed = m_source_pat(non_zero_inds);
        m_adapt_pat_processed = m_adapt_pat(non_zero_inds);
        m_target_pat_processed = m_target_pat(non_zero_inds);
        
        source_pat_error = compute_mse(g_t_pat_processed, m_source_pat_processed);
        
        adapt_pat_error = compute_mse(g_t_pat_processed, m_adapt_pat_processed);
        
        target_pat_error = compute_mse(g_t_pat_processed, m_target_pat_processed);
        
        % assesmble arrays 
        source_m_dim = size(m_source_pat); 
        source_m_dim = source_m_dim(1); 
        ID_all = [ID_all; ones(source_m_dim,1)*pat]; 
        g_t = [g_t; g_t_patient];
        m_adapt = [m_adapt; m_adapt_pat]; 
        m_target = [m_target; m_target_pat]; 
        s_adapt = [s_adapt; s_adapt_pat];
        s_target = [s_target; s_target_pat]; 
        tst_inds = [tst_inds; tst_ind_patient]; 
        
        if model_num == 2 
            error = [error; base_pat_error source_pat_error adapt_pat_error target_pat_error];
        else 
            error = [error; source_pat_error adapt_pat_error target_pat_error];
        end 
        
    end 
    
    % assemble arrays 
    m_source = [m_source; m_s]; 
    s_source = [s_source; s_s];
    
end  

% assemble arrays 
if model_num == 2 
    results = [ID_all g_t m_base m_source s_source m_adapt s_adapt m_target s_target tst_inds]; 
else 
    results = [ID_all g_t m_source s_source m_adapt s_adapt m_target s_target tst_inds]; 
end 

disp(size(error))
disp(size(ids_original))

num_folds = end_fold - start_fold + 1;

errors = [ids_original(1:10*num_folds,1:end) error]; 

% write results to CSV 
csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/final_results.csv'], results);
csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/final_errors.csv'], errors);

% assemble results by group and write to CSV 
if start_fold == 1 & end_fold == 10 % if we span all 10 folds... 
    % generate errors by group 
    g1_ids_original = [21.0, 23.0, 31.0, 72.0, 89.0, 120.0, 130.0, 172.0, 186.0, 257.0, 260.0, 295.0, 298.0, 301.0, 359.0, 382.0, 413.0, 419.0, 441.0, 610.0, 618.0, 685.0, 934.0, 1232.0, 1261.0];
    g2_ids_original = [107.0, 127.0, 150.0, 169.0, 200.0, 307.0, 384.0, 454.0, 545.0, 546.0, 644.0, 668.0, 679.0, 722.0, 741.0, 800.0, 919.0, 1045.0, 1072.0, 1122.0, 1155.0, 1187.0, 1246.0, 1269.0, 1300.0, 1414.0, 1418.0];
    g3_ids_original = [51.0, 61.0, 108.0, 112.0, 123.0, 126.0, 135.0, 142.0, 214.0, 256.0, 259.0, 269.0, 276.0, 331.0, 361.0, 376.0, 378.0, 388.0, 539.0, 548.0, 626.0, 649.0, 658.0, 671.0, 698.0, 729.0, 752.0, 778.0, 830.0, 835.0, 869.0, 887.0, 906.0, 952.0, 972.0, 984.0, 985.0, 994.0, 1078.0, 1097.0, 1098.0, 1123.0, 1186.0, 1268.0, 1318.0, 1346.0, 1351.0, 1427.0];
    
    % get indices of group errors 
    g1_error_inds = find(ismember(errors(1:end, 1:1), g1_ids_original));
    g2_error_inds = find(ismember(errors(1:end, 1:1), g2_ids_original));
    g3_error_inds = find(ismember(errors(1:end, 1:1), g3_ids_original));
    
    g1_errors = errors(g1_error_inds, 1:end); 
    g2_errors = errors(g2_error_inds, 1:end); 
    g3_errors = errors(g3_error_inds, 1:end); 
    
    csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/g1_final_errors.csv'], g1_errors);
    csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/g2_final_errors.csv'], g2_errors);
    csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/g3_final_errors.csv'], g3_errors);
end 
end 