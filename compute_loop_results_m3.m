function compute_loop_results_m3(fold_list, loop_count)
% use for model 3 
model_num = 3; 

disp('Computing loop results...')

% get current directory 
currentFolder = pwd; 

% get list of test patient IDs for each fold 
all_ids = csvread([currentFolder '/Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv']);

% hardcoded patients for each fold 
% indices 
g1_ids{1} = [21.0, 23.0, 31.0, 72.0, 89.0, 120.0, 130.0];
g1_ids{2} = [172.0, 186.0, 257.0, 260.0, 295.0, 298.0];
g1_ids{3} = [301.0, 359.0, 382.0, 413.0, 419.0, 441.0];
g1_ids{4} = [610.0, 618.0, 685.0, 934.0, 1232.0, 1261.0];

g2_ids{1} =  [107.0, 127.0, 150.0, 169.0, 200.0, 307.0, 384.0];
g2_ids{2} =  [454.0, 545.0, 546.0, 644.0, 668.0, 679.0, 722.0];
g2_ids{3} =  [741.0, 800.0, 919.0, 1045.0, 1072.0, 1122.0, 1155.0];
g2_ids{4} =  [1187.0, 1246.0, 1269.0, 1300.0, 1414.0, 1418.0];

g3_ids{1} = [51.0,61.0,108.0,112.0,123.0,126.0,135.0,142.0,214.0,256.0,259.0,269.0];
g3_ids{2} = [276.0, 331.0, 361.0, 376.0, 378.0, 388.0, 539.0, 548.0, 626.0, 649.0, 658.0, 671.0];
g3_ids{3} = [698.0, 729.0, 752.0, 778.0, 830.0, 835.0, 869.0, 887.0, 906.0, 952.0, 972.0, 984.0];
g3_ids{4} = [985.0, 994.0, 1078.0, 1097.0, 1098.0, 1123.0, 1186.0, 1268.0, 1318.0, 1346.0, 1351.0, 1427.0];

ids{1} = g1_ids;
ids{2} = g2_ids;
ids{3} = g3_ids; 

g1_ids_original = [21.0, 23.0, 31.0, 72.0, 89.0, 120.0, 130.0, 172.0, 186.0, 257.0, 260.0, 295.0, 298.0, 301.0, 359.0, 382.0, 413.0, 419.0, 441.0, 610.0, 618.0, 685.0, 934.0, 1232.0, 1261.0];
g2_ids_original = [107.0, 127.0, 150.0, 169.0, 200.0, 307.0, 384.0, 454.0, 545.0, 546.0, 644.0, 668.0, 679.0, 722.0, 741.0, 800.0, 919.0, 1045.0, 1072.0, 1122.0, 1155.0, 1187.0, 1246.0, 1269.0, 1300.0, 1414.0, 1418.0];
g3_ids_original = [51.0, 61.0, 108.0, 112.0, 123.0, 126.0, 135.0, 142.0, 214.0, 256.0, 259.0, 269.0, 276.0, 331.0, 361.0, 376.0, 378.0, 388.0, 539.0, 548.0, 626.0, 649.0, 658.0, 671.0, 698.0, 729.0, 752.0, 778.0, 830.0, 835.0, 869.0, 887.0, 906.0, 952.0, 972.0, 984.0, 985.0, 994.0, 1078.0, 1097.0, 1098.0, 1123.0, 1186.0, 1268.0, 1318.0, 1346.0, 1351.0, 1427.0];

ids_original{1} = g1_ids_original;
ids_original{2} = g2_ids_original;
ids_original{3} = g3_ids_original;

% for each fold (iterates over list of fold numbers): 
for fold = fold_list
    
    disp(['Fold ' num2str(fold) ' in progress...']) 
    
    % find folder 
    foldFolder = [currentFolder '/kgp_results/m' num2str(model_num) '_mat/fold_' num2str(fold)]; 
    
    % for each group (iterates over each group): 
    for group = 1:3 
        
        disp(['Group ' num2str(group) ' in progress...']) 
        
        % get test patient IDs for this group & fold  
        fold_group_test_ids = ids{group}{fold}; 
        fold_group_train_ids = setdiff(ids_original{group}, fold_group_test_ids); 
        

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
            for pat = fold_group_test_ids
                % load patient variables 
                x_a = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_x_a.mat']); 
                y_a = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_y_a.mat']); 
                x_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_x_s.mat']); 
                y_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_y_s.mat']); 
                xtest = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_xtest.mat']); 
                m_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_m_s.mat']); 
                s_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_s_s.mat']); 

                ls = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_ls.mat']);
                mul = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_mul.mat']);
                var = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_var.mat']);
                sn2 = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_sn2.mat']);

                K_ts_star_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_K_ts_star_all.mat']);
                K_tt_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_K_tt_all.mat']);
                K_t_star_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_K_t_star_all.mat']);
                L = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_L.mat']);
                alpha = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_alpha.mat']);

                k_star_star_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_k_star_star_all.mat']);

                g_t_patient = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/testing/id_' num2str(pat) '_g_t_patient.mat']);

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
                pat_ind = find(fold_group_test_ids' == pat); 
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

                % get mean and variance predictions 
                [m_adapt_pat, s_adapt_pat] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha);
                [m_target_pat, s_target_pat] = call_target(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all);

                % get error 
                pat_ind = find(fold_group_test_ids' == pat); 
                m_source_pat = m_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
                source_pat_error = compute_mse(g_t_patient, m_source_pat);

                adapt_pat_error = compute_mse(g_t_patient, m_adapt_pat);

                target_pat_error = compute_mse(g_t_patient, m_target_pat);
                
                % assesmble arrays 
                sgp_iter_test_error = [sgp_iter_test_error; source_pat_error];
                pgp_iter_test_error = [pgp_iter_test_error; adapt_pat_error];
                tgp_iter_test_error = [tgp_iter_test_error; target_pat_error];

            end 

            % for each train patient: 
            for pat = fold_group_train_ids
                % load patient variables 
                x_a = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_x_a.mat']); 
                y_a = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_y_a.mat']); 
                x_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_x_s.mat']); 
                y_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_y_s.mat']); 
                xtest = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_xtest.mat']); 
                m_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_m_s.mat']); 
                s_s = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_s_s.mat']); 

                ls = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_ls.mat']);
                mul = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_mul.mat']);
                var = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_var.mat']);
                sn2 = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_sn2.mat']);

                K_ts_star_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_K_ts_star_all.mat']);
                K_tt_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_K_tt_all.mat']);
                K_t_star_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_K_t_star_all.mat']);
                L = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_L.mat']);
                alpha = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_alpha.mat']);

                k_star_star_all = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_k_star_star_all.mat']);

                g_t_patient = load([foldFolder '/g' num2str(group) '_loop_' num2str(iter) '/training/id_' num2str(pat) '_g_t_patient.mat']);

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
                pat_ind = find(fold_group_train_ids' == pat); 
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

                % get mean and variance predictions 
                [m_adapt_pat, s_adapt_pat] = call_adapt(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_ts_star_all, K_tt_all, K_t_star_all, L, alpha);
                [m_target_pat, s_target_pat] = call_target(x_a, y_a, x_s, y_s, xtest, m_source_pat, s_source_pat, ls, mul, var, sn2, K_t_star_all, k_star_star_all, K_tt_all);

                % get error 
                pat_ind = find(fold_group_train_ids' == pat); 
                m_source_pat = m_s(21*(pat_ind-1)+1:21*pat_ind,1:end); 
                source_pat_error = compute_mse(g_t_patient, m_source_pat);

                adapt_pat_error = compute_mse(g_t_patient, m_adapt_pat);

                target_pat_error = compute_mse(g_t_patient, m_target_pat);

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

        % assemble arrays 
        errors = [sgp_fold_train_error sgp_fold_test_error pgp_fold_train_error pgp_fold_test_error tgp_fold_train_error tgp_fold_test_error];

        % write results to CSV 
        csvwrite([currentFolder '/kgp_results/model_' num2str(model_num) '/g' num2str(group) '_f' num2str(fold) '_iter_errors.csv'], errors);
    end 
end 