% compute model 0 results 
disp('Computing model 0 results...')
compute_final_results(0, 1, 10) % for fold 1 to 10 

% compute model 2 results 
disp('Computing model 2 results...')
loop_count = 15
compute_loop_results(2, 1, 1, loop_count) % for fold 1 only 
compute_final_results(2, 1, 1) % for fold 1 only 

% compute model 3 results 
disp('Computing model 3 results...')
loop_count = 10 
compute_loop_results_m3([1], loop_count) % for fold 1 only 
compute_final_results_m3([1], 3) % for fold 1 only 