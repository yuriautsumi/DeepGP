function compute_loop_results(model_num, start_fold, end_fold, loop_count)
% use for model 2 

disp('Computing loop results...')

% get current directory 
currentFolder = pwd; 

% get list of test patient IDs for each fold 
ids_original = csvread([currentFolder '/Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv']);
ids = reshape(ids_original, 10, 10); 

% for each fold: 
% for fold = 1:2 % 1 to 10 folds 
for fold = start_fold:end_fold % 1 to 10 folds 
    
    disp(['Fold ' num2str(fold) ' in progress...']) 
    
    % find folder 
    foldFolder = [currentFolder '/kgp_results/m' num2str(model_num) '_mat/fold_' num2str(fold)]; 
    
    % get patients 
    fold_test_ids = ids(1:end, fold);
    fold_train_ids = setdiff(ids_original, fold_test_ids); 
    
    % initialize arrays, each list end with loop_countx1
    sgp_fold_train_error = [];
    sgp_fold_test_error = []; 
    pgp_fold_train_error = [];
    pgp_fold_test_error = []; 
    tgp_fold_train_error = [];
    tgp_fold_test_error = []; 
    
    % for each iteration: 
    for iter = 1:loop_count 
        
            disp(['Iteration ' num2str(iter) ' in progress...'])
        
        sgp_iter_test_error = [];
        sgp_iter_train_error = [];        
        pgp_iter_test_error = [];
        pgp_iter_train_error = [];
        tgp_iter_test_error = [];
        tgp_iter_train_error = [];
        
        % for each test patient: 
        for pat = fold_test_ids'
            % load patient variables 
            x_a = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_x_a.mat']); 
            y_a = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_y_a.mat']); 
            x_s = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_x_s.mat']); 
            y_s = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_y_s.mat']); 
            xtest = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_xtest.mat']); 
            m_s = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_m_s.mat']); 
            s_s = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_s_s.mat']); 

            ls = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_ls.mat']);
            mul = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_mul.mat']);
            var = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_var.mat']);
            sn2 = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_sn2.mat']);

            K_ts_star_all = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_K_ts_star_all.mat']);
            K_tt_all = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_K_tt_all.mat']);
            K_t_star_all = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_K_t_star_all.mat']);
            L = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_L.mat']);
            alpha = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_alpha.mat']);

            k_star_star_all = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_k_star_star_all.mat']);

            g_t_patient = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_g_t_patient.mat']);
            
            tst_ind_patient = load([foldFolder '/loop_' num2str(iter) '/testing/id_' num2str(pat) '_tst_ind_patient.mat']);
            
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
            pat_ind = find(fold_test_ids' == pat); 
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
            
            g_t_patient = g_t_patient.g_t_patient; 
            tst_ind_patient = tst_ind_patient.tst_ind_patient; 

            % get mean and variance predictions 
            [m_adapt_pat, s_adapt_pat] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha);
            [m_target_pat, s_target_pat] = call_target(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all);

            % get error 
            pat_ind = find(fold_test_ids' == pat); 
            m_source_pat = m_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
            
            non_zero_inds = find(tst_ind_patient~=0);
            g_t_pat_processed = g_t_patient(non_zero_inds); 
            m_source_pat_processed = m_source_pat(non_zero_inds);
            m_adapt_pat_processed = m_adapt_pat(non_zero_inds);
            m_target_pat_processed = m_target_pat(non_zero_inds);

            source_pat_error = compute_mse(g_t_pat_processed, m_source_pat_processed);

            adapt_pat_error = compute_mse(g_t_pat_processed, m_adapt_pat_processed);

            target_pat_error = compute_mse(g_t_pat_processed, m_target_pat_processed);

            % assesmble arrays 
            sgp_iter_test_error = [sgp_iter_test_error; source_pat_error];
            pgp_iter_test_error = [pgp_iter_test_error; adapt_pat_error];
            tgp_iter_test_error = [tgp_iter_test_error; target_pat_error];

        end 
        
        % for each train patient: 
        for pat = fold_train_ids'
            % load patient variables 
            x_a = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_x_a.mat']); 
            y_a = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_y_a.mat']); 
            x_s = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_x_s.mat']); 
            y_s = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_y_s.mat']); 
            xtest = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_xtest.mat']); 
            m_s = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_m_s.mat']); 
            s_s = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_s_s.mat']); 

            ls = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_ls.mat']);
            mul = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_mul.mat']);
            var = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_var.mat']);
            sn2 = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_sn2.mat']);

            K_ts_star_all = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_K_ts_star_all.mat']);
            K_tt_all = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_K_tt_all.mat']);
            K_t_star_all = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_K_t_star_all.mat']);
            L = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_L.mat']);
            alpha = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_alpha.mat']);

            k_star_star_all = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_k_star_star_all.mat']);

            g_t_patient = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_g_t_patient.mat']);
            
            tr_ind_patient = load([foldFolder '/loop_' num2str(iter) '/training/id_' num2str(pat) '_tr_ind_patient.mat']);
            
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
            pat_ind = find(fold_train_ids' == pat); 
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
            
            g_t_patient = g_t_patient.g_t_patient; 
            tr_ind_patient = tr_ind_patient.tr_ind_patient;
            
            % get mean and variance predictions 
            [m_adapt_pat, s_adapt_pat] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha);
            [m_target_pat, s_target_pat] = call_target(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all);
            
            % get error 
            pat_ind = find(fold_train_ids' == pat); 
            m_source_pat = m_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
            
            non_zero_inds = find(tst_ind_patient~=0);
            g_t_pat_processed = g_t_patient(non_zero_inds); 
            m_source_pat_processed = m_source_pat(non_zero_inds);
            m_adapt_pat_processed = m_adapt_pat(non_zero_inds);
            m_target_pat_processed = m_target_pat(non_zero_inds);
            
            source_pat_error = compute_mse(g_t_pat_processed, m_source_pat_processed);

            adapt_pat_error = compute_mse(g_t_pat_processed, m_adapt_pat_processed);

            target_pat_error = compute_mse(g_t_pat_processed, m_target_pat_processed);
            
            % assesmble arrays 
            sgp_iter_train_error = [sgp_iter_train_error; source_pat_error];
            pgp_iter_train_error = [pgp_iter_train_error; adapt_pat_error];
            tgp_iter_train_error = [tgp_iter_train_error; target_pat_error];            
        end 
        
        % append average error to appropriate list 
        sgp_fold_train_error = [sgp_fold_train_error; mean(sgp_iter_train_error)]; 
        sgp_fold_test_error = [sgp_fold_test_error; mean(sgp_iter_test_error)]; 
        pgp_fold_train_error = [pgp_fold_train_error; mean(pgp_iter_train_error)];
        pgp_fold_test_error = [pgp_fold_test_error; mean(pgp_iter_test_error)]; 
        tgp_fold_train_error = [tgp_fold_train_error; mean(tgp_iter_train_error)];
        tgp_fold_test_error = [tgp_fold_test_error; mean(tgp_iter_test_error)]; 
        
    end 
    
end  

% assemble arrays 
errors = [sgp_fold_train_error sgp_fold_test_error pgp_fold_train_error pgp_fold_test_error tgp_fold_train_error tgp_fold_test_error];

% write results to CSV 
csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/iter_errors.csv'], errors);
end 