clear all;
close all;
clc;


load CSIQ_data.mat;

   load Subjective_ScoreCSIQ.mat;

%%%% SVR PARAMETERS ON CSIQ DATABASE
bestp = 1;   %%% bestp is one para that depends on DMOS range. For all databases, we scale the dmos into range to [0 100],
%%% and fix the p para as 1.

 bestc =2^9;  bestg =2^(-4);   %%% SVR PARAS ON 08 DB. These two paras are fixed for each db.
for x = 1:1
    
  
    features = [];
   
    
    features = [features data];
    
    if size(mos,1) == 1;
        mos=mos';
    end
    
    [mos_map,psmap] = mapminmax(mos',0,100);
    mos_map = mos_map';
    
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p ',num2str(bestp)];
    
    rts = 1000;
    
    results = zeros(rts, 4);
  
  
    for rt=1:rts
        type_idx = randperm( max(org_label(:)) );
        org_label_idx = type_idx(org_label);
        
        train_num = round(max(org_label(:))*0.8);
        train_data = features( org_label_idx<=train_num,:);
        train_mos = mos_map( org_label_idx<=train_num );
        
        test_data = features(org_label_idx>train_num,:);
        test_mos = mos( org_label_idx>train_num );
        
        [train_map, ps] = mapminmax(train_data',-1,1);
        train_data = train_map';
        test_map = mapminmax('apply',test_data',ps);
        test_data = test_map';
        
        train_scale = train_data; test_scale = test_data;
        svr_model = svmtrain(train_mos, train_scale, cmd);
        [pred_mos, accuracy, prob_esti] = svmpredict(test_mos, test_scale, svr_model);
        
        pred_mos_map = mapminmax('reverse',pred_mos',psmap);
        pred_mos = pred_mos_map';
        
       % srcc = IQAPerformance(pred_mos(:),test_mos(:),'s');
      srcc= corr(pred_mos(:), test_mos(:), 'type', 'spearman');
        krcc = corr(pred_mos(:),test_mos(:),'type','Kendall');
        cc = IQAPerformance(pred_mos(:),test_mos(:),'p');
        rmse = IQAPerformance(pred_mos(:),test_mos(:),'e');
        
        
        results(rt,1) = srcc; results(rt,2) = krcc;
        results(rt,3) = cc; results(rt,4) = rmse;
        
        
       
        clc;
        
    end
    frlist=  median(results);    %% overall performance on whole database
 
    frlist1 =  mean(results);    %% overall performance on whole database
  
end