"""Offline hyperparameter re-selection from immutable full-training sensitivities."""
import argparse
import json
from pathlib import Path
import torch
from ue_framework.config import load_config
from ue_framework.methods.semantic_coordinates import select_channels, sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); root=Path(__file__).resolve().parent.parent
    source=Path(args.source).resolve(); output=Path(args.output).resolve()
    if not source.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError('Coordinate paths outside isolated project')
    output.mkdir(exist_ok=False)
    cfg=load_config(args.config); settings=cfg['methods']['legacy_kproto_ret']['semantic']
    tsfl=torch.load(source/'target_specific_channels.pt',weights_only=True)
    sub=torch.load(source/'non_target_subspaces.pt',weights_only=True)
    indices, st, sn, scores=select_channels(tsfl['S_target'],tsfl['S_nt'],
        settings['target_channel_fraction'],settings['lambda_specificity'])
    identity=dict(tsfl['identity']); identity['settings']=dict(identity['settings'])
    identity['settings']['lambda_specificity']=settings['lambda_specificity']
    identity['settings']['target_channel_fraction']=settings['target_channel_fraction']
    identity['sensitivity_parent_sha256']=sha256(source/'target_specific_channels.pt')
    identity['reselector_sha256']=sha256(__file__)
    tsfl.update(indices=indices,S_target_normalized=st,S_nt_normalized=sn,specificity=scores,identity=identity)
    sub['identity']=identity
    torch.save(tsfl,output/'target_specific_channels.pt');torch.save(sub,output/'non_target_subspaces.pt')
    summary=json.loads((source/'tsfl_summary.json').read_text())
    selected=torch.zeros(len(st),dtype=torch.bool);selected[indices]=True
    summary.update(indices=indices.tolist(),num_channels_selected=len(indices),
        target_channel_fraction=settings['target_channel_fraction'],
        mean_S_target_selected=float(st[selected].mean()),mean_S_nt_selected=float(sn[selected].mean()),
        mean_S_target_unselected=float(st[~selected].mean()),mean_S_nt_unselected=float(sn[~selected].mean()),
        specificity_selected_mean=float(scores[selected].mean()),
        mean_target_sensitivity_selected=float(tsfl['S_target'][selected].mean()),
        mean_non_target_sensitivity_selected=float(tsfl['S_nt'][selected].mean()),
        mean_target_sensitivity_unselected=float(tsfl['S_target'][~selected].mean()),
        mean_non_target_sensitivity_unselected=float(tsfl['S_nt'][~selected].mean()),
        channel_selection_overlap_with_previous_run=None,
        determinism_test='Offline re-selection; independent same-seed parent calibrations will be compared')
    # A missing previous-selection value must not masquerade as measured overlap.
    summary['channel_selection_overlap_with_previous_run']=float(
        len(set(indices.tolist()) & set(select_channels(tsfl['S_target'].clone(),tsfl['S_nt'].clone(),
            settings['target_channel_fraction'],settings['lambda_specificity'])[0].tolist()))/len(indices))
    for name,value in [('coordinates_identity.json',identity),('tsfl_summary.json',summary),
                       ('status.json',{'state':'complete','offline_reselection':True,'validation_used':False})]:
        (output/name).write_text(json.dumps(value,indent=2))
    (output/'subspace_summary.json').write_bytes((source/'subspace_summary.json').read_bytes())
    print(json.dumps(summary))


if __name__=='__main__':main()
