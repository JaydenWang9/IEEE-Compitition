"""Export the submission repository's frozen horizon Task2 model to official keys.

No model fitting, threshold selection, future state reconstruction, or label use.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();repo=args.repository.resolve();package=args.package.resolve();data=args.data_root.resolve();work=args.work.resolve()
    assert not work.is_relative_to(data)
    work.mkdir(parents=True,exist_ok=True)
    tasks=repo/'experiments/task2';horizon=tasks/'best_overall_082723_horizon_20261031'
    sys.path.insert(0,str(horizon/'code'))
    from hybrid_horizon import Hybrid,verify_manifest
    import numpy as np,pandas as pd
    verified=verify_manifest(horizon);model=Hybrid(horizon)
    sys.path.insert(0,str(tasks/'r127_reproduction_20261027'))
    from repro_common import setup
    parent,_,_,audit=setup(package,data,work/'runtime',True,extra_allowed=(tasks,work))
    model.attach_parent(parent)
    # The repository adapter loads every new head before the immutable R127
    # archive-only native-model guard is installed by setup().
    from task2q.data import PANELS,released_sample
    from task2q.submission import validate_queue
    source=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    signature={'horizon_profile_sha256':source(horizon/'profile.json'),
               'horizon_manifest_sha256':source(horizon/'manifest.json'),
               'r127_manifest_sha256':source(package/'archive-manifest.json'),
               'exporter_sha256':source(__file__),'data_root':str(data),'smoke':args.smoke}
    lock=work/'input-signature.json'
    if lock.exists():assert json.loads(lock.read_text(encoding='utf-8'))==signature,'Inputs changed; use a new work directory'
    else:lock.write_text(json.dumps(signature,indent=2),encoding='utf-8')
    checks=[];all_parts={};start=time.time()
    for split in ['validation','private']:
        parts=[]
        for panel in PANELS:
            folder=data/'task2'/panel/split
            index=pd.read_csv(folder/'window_index.csv',dtype={'window_id':str})
            template=pd.read_csv(folder/'sample_submission_queue.csv',dtype={'window_id':str,'link_id':str})
            if args.smoke:
                index=index.groupby('condition',sort=False).head(1)
                template=template[template.window_id.isin(index.window_id)].copy()
            cache=work/(split+'_'+panel+'.csv')
            if cache.exists():
                frame=pd.read_csv(cache,dtype={'window_id':str,'link_id':str})
                validate_queue(frame,template);parts.append(frame);continue
            panel_parts=[];seen=set()
            for row in index.itertuples():
                s=released_sample(data,panel,split,row.window_id)
                probability=model.probability(s);binary=model.decode(s,probability)
                assert probability.shape==(6,len(s['network']['link_ids'])) and binary.shape==probability.shape
                assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                if s['condition'] not in seen:
                    other=copy.deepcopy(s);other['y']=1-other['y'];other['m']=~other['m'];other['label_mask']=~other['label_mask'];other['soft_y']=np.full_like(other['soft_y'],77);other['speed_y']=np.full_like(other['speed_y'],999);other['has_future_queue']=True
                    np.testing.assert_array_equal(probability,model.probability(other))
                    checks.append({'panel':panel,'split':split,'condition':int(s['condition']),'future_label_mutation_no_effect':True})
                    seen.add(s['condition'])
                keys=template[template.window_id==row.window_id].copy()
                lookup={str(link):i for i,link in enumerate(s['network']['link_ids'])}
                horizon_index=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(int)-1
                node=keys.link_id.map(lookup).to_numpy()
                assert ((horizon_index>=0)&(horizon_index<6)).all() and not pd.isna(node).any()
                keys['queue_pred']=binary[horizon_index,node.astype(int)].astype(int);keys['panel']=panel
                panel_parts.append(keys)
            frame=pd.concat(panel_parts,ignore_index=True);validate_queue(frame,template)
            frame.to_csv(cache,index=False);parts.append(frame)
            print(f'{split} {panel}: {len(frame)} rows, {len(index)} official windows; {time.time()-start:.1f}s',flush=True)
        result=pd.concat(parts,ignore_index=True)
        if not args.smoke:assert len(result)==87000
        assert not result.duplicated(['panel','window_id','timestamp','link_id']).any()
        assert result.queue_pred.isin([0,1]).all()
        result.to_csv(work/('queue-'+split+'.csv'),index=False);all_parts[split]=len(result)
    report={'passed':True,'rows_by_split':all_parts,'total_rows':sum(all_parts.values()),
            'complete_official_templates':not args.smoke,'published_horizon_files_verified':verified,
            'source_repository':'chengweiliu669-pixel/25-day-a-Big-Dream-submission',
            'source_commit':'d4a4810cf4add1cec039878eb87b9ae48c684b73',
            'future_label_causality_checks':checks,'no_Task1_predictions_used':True,
            'no_new_training_or_threshold_selection':True,'official_S_queue':None,
            'source_signature':signature,'seconds':time.time()-start}
    (work/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'total_rows':report['total_rows'],'complete':not args.smoke}),flush=True)


if __name__=='__main__':main()
