#!/usr/bin/env python3
"""Leakage-aware 15-feature HKJC WIN ranking and QP-pair diagnostic.

- Reuses the audited past-only 15-feature frame builder.
- Splits complete race dates chronologically (70/15/15).
- Compares the deployed legacy WIN classifier, an unweighted classifier,
  a within-race LambdaRank model, and a validation-calibrated race softmax.
- Fits a PLACE classifier on the same training split to evaluate top-two QP
  pair hit rate (both selected horses officially placed).
- Saves candidate models separately; never overwrites the production model.

The source history contains final WIN odds, not timestamped snapshots. Thus the
holdout estimates ranking with that historical market input, not performance
at a specific earlier live-odds timestamp. Replace odds with archived snapshots
before making a time-specific betting/ROI claim.
"""
from __future__ import annotations
import argparse, json, sys, warnings
import subprocess
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

UPLOAD = Path(__file__).resolve().parent
REPO = UPLOAD
TRAIN_SCRIPT = UPLOAD / 'train_hkjc_place_15feature_time_split.py'
sys.path.insert(0, str(UPLOAD))
from train_hkjc_place_15feature_time_split import (  # noqa: E402
    FEATURE_COLS_15, MODEL_PARAMS, build_training_frame, chronological_masks, read_csv_robust,
)

RANKER_PARAMS = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [1, 3],
    'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 6,
    'num_leaves': 15, 'min_child_samples': 35, 'reg_lambda': 1.0,
    'random_state': 42, 'verbosity': -1,
}
PAIR_FEATURES = [f"pair_{stat}_{col}" for col in FEATURE_COLS_15 for stat in ("sum", "absdiff", "min", "max")]
PAIR_RANKER_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3],
    "n_estimators": 2000, "learning_rate": 0.02, "max_depth": 6,
    "num_leaves": 15, "min_child_samples": 40, "reg_lambda": 1.0,
    "random_state": 43, "verbosity": -1,
}


def groups_in_order(frame: pd.DataFrame) -> list[int]:
    return frame.groupby('賽事編號', sort=False, observed=True).size().astype(int).tolist()


def race_rank_metrics(frame: pd.DataFrame, scores: np.ndarray, name: str) -> dict:
    x = frame[['賽事編號', '馬號', '名次', '_is_win', 'target_place', '獨贏賠率']].copy().reset_index(drop=True)
    x['_score'] = np.asarray(scores, dtype=float)
    win_ranks, top1, top3, winner_probs, winner_labels = [], [], [], [], []
    race_sums=[]
    for _, g in x.groupby('賽事編號', sort=False, observed=True):
        # Select exactly one horse per race; stable tie break by horse number.
        g = g.sort_values(['_score', '馬號'], ascending=[False, True], kind='mergesort')
        winner = g.loc[g['_is_win'].eq(1)]
        if len(winner) != 1:
            continue
        winner_index=winner.index[0]
        rank=int(np.flatnonzero(g.index.to_numpy()==winner_index)[0])+1
        win_ranks.append(rank)
        top1.append(int(rank==1)); top3.append(int(rank<=3))
        # Markets in this source are final odds; normalized per race as benchmark.
        market = pd.to_numeric(g['獨贏賠率'], errors='coerce').to_numpy(float)
        market_raw=np.where(np.isfinite(market)&(market>1),1/market,0.0)
        market_prob=market_raw/market_raw.sum() if market_raw.sum()>0 else np.full(len(g),1/len(g))
        winner_idx=int(np.flatnonzero(g['_is_win'].to_numpy()==1)[0])
        race_sums.append(float(np.sum(np.exp(np.clip(g['_score'].to_numpy()-np.max(g['_score']),-700,0)))))
        winner_probs.append(market_prob[winner_idx]); winner_labels.append(winner_idx)
    return {
        'races': int(len(win_ranks)), 'top1_win_rate': float(np.mean(top1)),
        'top1_wins': int(sum(top1)), 'top3_winner_coverage': float(np.mean(top3)),
        'top3_winner_races': int(sum(top3)), 'mean_winner_rank': float(np.mean(win_ranks)),
        'median_winner_rank': float(np.median(win_ranks)),
        'note': 'Exactly one horse selected per race; equal scores are resolved by ascending horse number.',
    }


def ranker_frame_metrics(frame: pd.DataFrame, scores: np.ndarray, temperature: float | None=None) -> dict:
    out=race_rank_metrics(frame,scores,'ranker')
    if temperature is None:
        return out
    x=frame[['賽事編號','馬號','_is_win']].copy().reset_index(drop=True);x['_s']=np.asarray(scores,float)
    nll=[]; briers=[]; prob_sums=[]
    for _,g in x.groupby('賽事編號',sort=False,observed=True):
        s=g['_s'].to_numpy(); z=(s-s.max())/temperature
        p=np.exp(np.clip(z,-700,0));p=p/p.sum()
        y=g['_is_win'].to_numpy(dtype=int)
        nll.append(-float(np.log(max(p[y==1][0],1e-15))))
        briers.append(float(np.sum((p-y)**2)))
        prob_sums.append(float(p.sum()))
    out.update({'temperature':float(temperature),'race_multiclass_log_loss':float(np.mean(nll)),
                'race_multiclass_brier_sum':float(np.mean(briers)),
                'probability_sum_mean':float(np.mean(prob_sums))})
    return out


def fit_temperature(frame: pd.DataFrame, scores: np.ndarray) -> float:
    x=frame[['賽事編號','_is_win']].copy().reset_index(drop=True);x['_s']=np.asarray(scores,float)
    races=[]
    for _,g in x.groupby('賽事編號',sort=False,observed=True):
        s=g['_s'].to_numpy(); y=g['_is_win'].to_numpy(dtype=int)
        races.append((s,y))
    def objective(log_t):
        t=float(np.exp(log_t)); losses=[]
        for s,y in races:
            z=(s-s.max())/t;p=np.exp(np.clip(z,-700,0));p/=p.sum()
            losses.append(-np.log(max(p[y==1][0],1e-15)))
        return float(np.mean(losses))
    result=minimize_scalar(objective,bounds=(np.log(0.03),np.log(100.0)),method='bounded')
    return float(np.exp(result.x))


def pair_metrics(frame: pd.DataFrame, place_prob: np.ndarray) -> dict:
    x=frame[['賽事編號','馬號','target_place','獨贏賠率']].copy().reset_index(drop=True)
    x['_p']=np.asarray(place_prob,float)
    result=[]
    for rid,g in x.groupby('賽事編號',sort=False,observed=True):
        top=g.sort_values(['_p','馬號'],ascending=[False,True],kind='mergesort').head(2)
        market=g.sort_values(['獨贏賠率','馬號'],ascending=[True,True],kind='mergesort').head(2)
        result.append({'race_id':rid,'place_pair_both':int(top['target_place'].eq(1).all()),
                       'place_pair_one_or_more':int(top['target_place'].eq(1).any()),
                       'market_pair_both':int(market['target_place'].eq(1).all()),
                       'market_pair_one_or_more':int(market['target_place'].eq(1).any()),
                       'place_pair_horses':'+'.join(top['馬號'].astype(str)),
                       'market_pair_horses':'+'.join(market['馬號'].astype(str))})
    return {'races':len(result),'model_top2_both_in_place':sum(r['place_pair_both'] for r in result),
            'model_top2_both_rate':float(np.mean([r['place_pair_both'] for r in result])),
            'model_top2_at_least_one_place':sum(r['place_pair_one_or_more'] for r in result),
            'market_top2_both_in_place':sum(r['market_pair_both'] for r in result),
            'market_top2_both_rate':float(np.mean([r['market_pair_both'] for r in result])),
            'race_level_rows':result}


def make_qp_pair_frame(horses: pd.DataFrame) -> pd.DataFrame:
    """Make every unordered horse pair within each race; relevance=both in PLACE."""
    from itertools import combinations
    records=[]
    for race_id,g in horses.groupby('賽事編號',sort=False,observed=True):
        g=g.reset_index(drop=True)
        vals=g[FEATURE_COLS_15].astype(float).to_numpy()
        nos=g['馬號'].astype(str).to_numpy()
        placed=g['target_place'].astype(int).to_numpy()
        dates=g['_race_date'].to_numpy()
        for i,j in combinations(range(len(g)),2):
            a,b=vals[i],vals[j]
            rec={'賽事編號':str(race_id),'_race_date':dates[i],
                 '馬號1':nos[i],'馬號2':nos[j],
                 'pair_target':int(placed[i] and placed[j])}
            for k,col in enumerate(FEATURE_COLS_15):
                av,bv=float(a[k]),float(b[k])
                rec[f'pair_sum_{col}']=av+bv
                rec[f'pair_absdiff_{col}']=abs(av-bv)
                rec[f'pair_min_{col}']=min(av,bv)
                rec[f'pair_max_{col}']=max(av,bv)
            records.append(rec)
    return pd.DataFrame.from_records(records)


def evaluate_pair_ranker(pair_frame: pd.DataFrame, score: np.ndarray) -> tuple[dict,pd.DataFrame]:
    x=pair_frame[['賽事編號','_race_date','馬號1','馬號2','pair_target']].copy().reset_index(drop=True)
    x['ranker_score']=np.asarray(score,float)
    top=[]
    for rid,g in x.groupby('賽事編號',sort=False,observed=True):
        chosen=g.sort_values(['ranker_score','馬號1','馬號2'],ascending=[False,True,True],kind='mergesort').head(1)
        top.append(chosen)
    selected=pd.concat(top,ignore_index=True)
    pair_hit=int(selected['pair_target'].sum())
    # Candidate pair-top-3 ticket coverage: whether any of three first pairs is the true QP.
    coverage=[]
    for _,g in x.groupby('賽事編號',sort=False,observed=True):
        head=g.sort_values(['ranker_score','馬號1','馬號2'],ascending=[False,True,True],kind='mergesort').head(3)
        coverage.append(int(head['pair_target'].max()))
    return ({'races':int(len(selected)),'best_pair_both_in_place':pair_hit,
             'best_pair_both_rate':float(pair_hit/len(selected)) if len(selected) else None,
             'top3_pairs_at_least_one_valid_qp':int(sum(coverage)),
             'top3_pair_coverage_rate':float(np.mean(coverage)) if coverage else None},selected)


def main():
    parser=argparse.ArgumentParser(description='以15大past-only特徵設計及驗證WIN排序器/QP雙馬基準')
    parser.add_argument('--input',default=str(UPLOAD/'hkjc_all_seasons_features.csv'))
    parser.add_argument('--output-dir',default=str(UPLOAD/'win_ranker_redesign_output'))
    parser.add_argument('--legacy-model',default=str(REPO/'my_hkjc_model.pkl'))
    parser.add_argument('--legacy-place-model',default=str(REPO/'my_hkjc_place_model.pkl'))
    parser.add_argument('--legacy-win-form',choices=['fixed','dynamic'],default='fixed',help='既有WIN模型訓練Form，用於公平對照')
    parser.add_argument('--dynamic-form',action='store_true',help='與App的Dynamic form勾選狀態一致；預設為Fixed')
    parser.add_argument('--both-forms',action='store_true',help='分別訓練Fixed與Dynamic兩種輸入Form')
    parser.add_argument('--legacy-place-form',choices=['fixed','dynamic'],default='dynamic',help='GitHub現用PLACE模型訓練Form，用於防止錯配比較')
    args=parser.parse_args()
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    if args.both_forms and args.dynamic_form:
        parser.error('--both-forms不可與--dynamic-form同時使用')
    if args.both_forms:
        for mode,extra in [('fixed_form',[]),('dynamic_form',['--dynamic-form'])]:
            subdir=outdir/mode
            cmd=[sys.executable,str(Path(__file__).resolve()),'--input',args.input,'--output-dir',str(subdir),
                 '--legacy-model',args.legacy_model,'--legacy-place-model',args.legacy_place_model,
                 '--legacy-win-form',args.legacy_win_form,'--legacy-place-form',args.legacy_place_form,*extra]
            print(f'\n===== 開始 {mode} =====',flush=True);subprocess.run(cmd,check=True)
        return
    raw=read_csv_robust(Path(args.input));data,clean_stats=build_training_frame(raw,dynamic_form=args.dynamic_form)
    masks,train_cut,val_cut=chronological_masks(data)
    train=data.loc[masks['train']].copy();val=data.loc[masks['validation']].copy();test=data.loc[masks['test']].copy()
    Xtr,Xv,Xte=(d[FEATURE_COLS_15].astype(float) for d in [train,val,test])
    ytr,yv,yte=(d['_is_win'].astype(int) for d in [train,val,test])
    group_tr,group_v,group_te=(groups_in_order(d) for d in [train,val,test])
    assert sum(group_tr)==len(train) and sum(group_v)==len(val) and sum(group_te)==len(test)

    report={'input':str(Path(args.input).resolve()),'rows':len(data),'races':int(data['賽事編號'].nunique()),
            'date_range':[str(data['_race_date'].min().date()),str(data['_race_date'].max().date())],
            'cleaning':clean_stats,'split':{'train_end_exclusive':str(train_cut.date()),
            'validation_end_exclusive':str(val_cut.date()),'train_races':len(group_tr),
            'validation_races':len(group_v),'test_races':len(group_te),
            'test_dates':[str(test['_race_date'].min().date()),str(test['_race_date'].max().date())]},
            'warning':'Source history uses official final odds, not time-stamped pre-race snapshots; holdout does not represent a fixed live-odds decision time.'}

    # Existing production model: evaluate only with its matching input form.
    legacy_path=Path(args.legacy_model)
    requested_form='dynamic' if args.dynamic_form else 'fixed'
    legacy_win_scores=None
    if legacy_path.exists() and args.legacy_win_form==requested_form:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); legacy=joblib.load(legacy_path)
        legacy_scores=legacy.predict_proba(Xte)[:,1]
        legacy_win_scores=legacy_scores
        report['legacy_current_classifier']={**race_rank_metrics(test,legacy_scores,'legacy'),
            'mean_row_probability':float(np.mean(legacy_scores)),
            'median_race_probability_sum':float(pd.DataFrame({'race':test['賽事編號'].to_numpy(),'p':legacy_scores}).groupby('race').p.sum().median()),
            'sklearn_warnings':[str(w.message) for w in caught]}
        report['legacy_current_classifier']['row_metrics']={
            'auc':float(roc_auc_score(yte,legacy_scores)),
            'brier':float(np.mean((legacy_scores-yte.to_numpy())**2)),
            'log_loss':float(-np.mean(yte.to_numpy()*np.log(np.clip(legacy_scores,1e-15,1))+(1-yte.to_numpy())*np.log(np.clip(1-legacy_scores,1e-15,1))))}
    elif legacy_path.exists():
        report['legacy_current_classifier']={'not_compared':True,
            'reason':f'GitHub WIN模型設定為{args.legacy_win_form} form，本次輸入為{requested_form} form；避免不公平評估。'}

    # Binary classifier without class balancing, for calibration/ranking comparison.
    clf_params=dict(MODEL_PARAMS);clf_params['n_estimators']=2000;clf_params['class_weight']=None
    clf=lgb.LGBMClassifier(**clf_params)
    clf.fit(Xtr,ytr,eval_set=[(Xv,yv)],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.log_evaluation(0)])
    nclf=int(clf.best_iteration_ or clf_params['n_estimators'])
    clf_prob=clf.predict_proba(Xte,num_iteration=nclf)[:,1]
    report['unweighted_classifier']={**race_rank_metrics(test,clf_prob,'unweighted'),
        'best_iteration':nclf,'mean_row_probability':float(np.mean(clf_prob)),
        'median_race_probability_sum':float(pd.DataFrame({'race':test['賽事編號'].to_numpy(),'p':clf_prob}).groupby('race').p.sum().median()),
        'row_metrics':{'auc':float(roc_auc_score(yte,clf_prob)),'brier':float(np.mean((clf_prob-yte.to_numpy())**2)),
        'log_loss':float(-np.mean(yte.to_numpy()*np.log(np.clip(clf_prob,1e-15,1))+(1-yte.to_numpy())*np.log(np.clip(1-clf_prob,1e-15,1))))}}

    # LambdaRank directly optimizes within-race ordering; each race is one query.
    ranker=lgb.LGBMRanker(**RANKER_PARAMS)
    ranker.fit(Xtr,ytr,group=group_tr,eval_set=[(Xv,yv)],eval_group=[group_v],eval_at=[1,3],
               callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.log_evaluation(0)])
    n_rank=int(ranker.best_iteration_ or RANKER_PARAMS['n_estimators'])
    val_score=ranker.predict(Xv,num_iteration=n_rank)
    temp=fit_temperature(val,val_score)
    test_score=ranker.predict(Xte,num_iteration=n_rank)
    report['lambdarank']={**ranker_frame_metrics(test,test_score,temp),'best_iteration':n_rank,
                          'validation_temperature':temp}

    # Paired-place target, trained on same train/validation dates; evaluate top-two pair.
    yp_tr=train['target_place'].astype(int);yp_v=val['target_place'].astype(int)
    place_params=dict(MODEL_PARAMS);place_params['n_estimators']=2000;place_params['class_weight']=None
    place= lgb.LGBMClassifier(**place_params)
    place.fit(Xtr,yp_tr,eval_set=[(Xv,yp_v)],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.log_evaluation(0)])
    n_place=int(place.best_iteration_ or place_params['n_estimators'])
    place_prob=place.predict_proba(Xte,num_iteration=n_place)[:,1]
    report['qp_pair']=pair_metrics(test,place_prob)
    pair_rows=report['qp_pair'].pop('race_level_rows')
    report['qp_pair']['place_classifier_best_iteration']=n_place
    report['place_model_top1_metrics']=race_rank_metrics(test,place_prob,'place')
    pd.DataFrame(pair_rows).to_csv(outdir/'win_qp_pair_race_details.csv',index=False,encoding='utf-8-sig')
    incumbent_place_path=Path(args.legacy_place_model)
    if incumbent_place_path.exists() and args.legacy_place_form==requested_form:
        with warnings.catch_warnings(record=True) as place_warnings:
            warnings.simplefilter('always'); incumbent_place=joblib.load(incumbent_place_path)
        incumbent_place_prob=incumbent_place.predict_proba(Xte)[:,1]
        incumbent_pair=pair_metrics(test,incumbent_place_prob)
        incumbent_rows=incumbent_pair.pop('race_level_rows')
        report['legacy_place_model_qp_pair']={**incumbent_pair,
            'model_path':str(incumbent_place_path),'warnings':[str(w.message) for w in place_warnings]}
        pd.DataFrame(incumbent_rows).to_csv(outdir/'legacy_place_qp_pair_race_details.csv',index=False,encoding='utf-8-sig')
    elif incumbent_place_path.exists():
        report['legacy_place_model_qp_pair']={'not_compared':True,
            'reason':f"GitHub PLACE模型為{args.legacy_place_form} form，但本次候選輸入為{requested_form} form；避免不公平評估。"}

    # Direct QP pair ranker: each unordered pair is a candidate ticket and its
    # relevance is 1 only when both horses officially finish in a PLACE slot.
    pair_train=make_qp_pair_frame(train)
    pair_val=make_qp_pair_frame(val)
    pair_test=make_qp_pair_frame(test)
    pair_ranker=lgb.LGBMRanker(**PAIR_RANKER_PARAMS)
    pair_ranker.fit(
        pair_train[PAIR_FEATURES].astype(float),pair_train['pair_target'].astype(int),
        group=groups_in_order(pair_train),
        eval_set=[(pair_val[PAIR_FEATURES].astype(float),pair_val['pair_target'].astype(int))],
        eval_group=[groups_in_order(pair_val)],eval_at=[1,3],
        callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.log_evaluation(0)],
    )
    n_pair=int(pair_ranker.best_iteration_ or PAIR_RANKER_PARAMS['n_estimators'])
    pair_score=pair_ranker.predict(pair_test[PAIR_FEATURES].astype(float),num_iteration=n_pair)
    pair_result,pair_selected=evaluate_pair_ranker(pair_test,pair_score)
    report['direct_qp_pair_ranker']={**pair_result,'best_iteration':n_pair,
        'candidate_pair_rows':int(len(pair_test)),
        'positive_pair_rows':int(pair_test['pair_target'].sum()),
        'note':'每場直接排序所有無序雙馬組合；命中條件為兩馬均按該場有效入位席位入位。'}
    pair_test_out=pair_test[['賽事編號','_race_date','馬號1','馬號2','pair_target']].copy()
    pair_test_out['pair_ranker_score']=pair_score
    pair_test_out['pair_rank']=pair_test_out.groupby('賽事編號')['pair_ranker_score'].rank(method='first',ascending=False)
    pair_test_out.to_csv(outdir/'direct_qp_pair_holdout_predictions.csv',index=False,encoding='utf-8-sig')
    pair_selected.to_csv(outdir/'direct_qp_pair_top_pick_by_race.csv',index=False,encoding='utf-8-sig')

    # Simple market baseline using final win odds (not contemporaneous odds).
    market_score=1.0/pd.to_numeric(test['獨贏賠率'],errors='coerce').to_numpy(float)
    report['market_favorite']=race_rank_metrics(test,market_score,'market')

    # Holdout row-level table has all features and all candidate scores/ranks.
    scored=test[['賽事編號','_race_date','馬號','馬名','名次','_is_win','target_place','獨贏賠率']+FEATURE_COLS_15].copy().reset_index(drop=True)
    scored['legacy_win_probability']=legacy_win_scores if legacy_win_scores is not None else np.nan
    scored['unweighted_win_probability']=clf_prob
    scored['win_ranker_score']=test_score
    scored['ranker_temperature_probability']=np.nan
    for _,idx in scored.groupby('賽事編號',sort=False).groups.items():
        ix=np.asarray(list(idx),int);z=(test_score[ix]-np.max(test_score[ix]))/temp;p=np.exp(np.clip(z,-700,0));scored.loc[ix,'ranker_temperature_probability']=p/p.sum()
    scored['place_pair_candidate_probability']=place_prob
    scored['legacy_place_probability']=incumbent_place_prob if incumbent_place_path.exists() and args.legacy_place_form==requested_form else np.nan
    for col in ['legacy_win_probability','unweighted_win_probability','win_ranker_score','ranker_temperature_probability','place_pair_candidate_probability','legacy_place_probability']:
        if col!='win_ranker_score': scored[col.replace('_probability','_rank')]=scored.groupby('賽事編號')[col].rank(method='min',ascending=False)
    scored.to_csv(outdir/'win_qp_holdout_predictions.csv',index=False,encoding='utf-8-sig')

    (outdir/'win_qp_redesign_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    (outdir/'win_qp_pair_race_details.csv').to_csv(outdir/'win_qp_pair_race_details.csv',index=False) if False else None

    # Refit deployable candidates on all dated data after holdout evaluation.
    final_ranker=lgb.LGBMRanker(**{**RANKER_PARAMS,'n_estimators':n_rank})
    final_ranker.fit(data[FEATURE_COLS_15],data['_is_win'].astype(int),group=groups_in_order(data))
    joblib.dump(final_ranker,outdir/'my_hkjc_win_ranker_candidate.pkl')
    final_place=lgb.LGBMClassifier(**{**place_params,'n_estimators':n_place})
    final_place.fit(data[FEATURE_COLS_15],data['target_place'].astype(int))
    joblib.dump(final_place,outdir/'my_hkjc_place_pair_candidate.pkl')
    all_pairs=make_qp_pair_frame(data)
    final_pair_ranker=lgb.LGBMRanker(**{**PAIR_RANKER_PARAMS,'n_estimators':n_pair})
    final_pair_ranker.fit(all_pairs[PAIR_FEATURES].astype(float),all_pairs['pair_target'].astype(int),
                          group=groups_in_order(all_pairs))
    joblib.dump(final_pair_ranker,outdir/'my_hkjc_qp_pair_ranker_candidate.pkl')
    metadata={'win_model_type':'LightGBM LambdaRank','win_target':'within-race winner ordering; relevance=1 for winner, 0 otherwise',
              'place_model_type':'unweighted LightGBM binary classifier','qp_model_type':'LightGBM LambdaRank over unordered horse pairs',
              'qp_target':'both horses officially placed under HKJC runner-count rules',
              'features':FEATURE_COLS_15,'dynamic_form':bool(args.dynamic_form),'app_setting_match':('開啟Dynamic form' if args.dynamic_form else '關閉Dynamic form/Fixed'),
              'qp_pair_features':PAIR_FEATURES,
              'history_asof':'strictly earlier race dates; same-day results excluded',
              'final_fit_start_date':str(pd.to_datetime(data['_race_date']).min().date()),
              'final_fit_end_date':str(pd.to_datetime(data['_race_date']).max().date()),
              'final_fit_includes_holdout':True,
              'ranker_softmax_temperature':temp,'feature_count':len(FEATURE_COLS_15),
              'source_history_final_odds_warning':report['warning'],'holdout_metrics_file':'win_qp_redesign_metrics.json',
              'holdout_predictions_file':'win_qp_holdout_predictions.csv',
              'note':'Candidates are separate from production files. Holdout metrics estimate the training procedure; refit candidate includes all dated source history.'}
    (outdir/'candidate_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('\n候選輸出：',outdir)
    print('舊版 production model untouched:',legacy_path)

if __name__=='__main__': main()
