"""Read-only model inference followed by isolated, offline structure evaluation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.kuramoto import label_matrices
from .metrics import (association_baselines, binary_metrics, family_ci, finite_mean,
                      matrix_comparison, score_matrices, symmetric)
from .model.attention_probe import AttentionProbe
from .utils import (environment, fresh_directory, get_device, read_config, run_id,
                    seed_all, sha256_file, write_csv, write_json)

METRICS = ("ap", "auroc", "precision_at_k", "density", "positive_score_mean", "negative_score_mean")
TARGETS = ("pairwise", "projected-triad", "any-interaction", "triad-only-subset")


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def aggregate(rows, samples):
    keys = ("stage", "method", "target", "family_id", "topology_id", "variant")
    topology = []
    for key, group in grouped(rows, keys).items():
        row = dict(zip(keys,key))
        row.update(run_id=group[0]["run_id"], n_trajectories=len(group),
                   n_valid_ap=sum(np.isfinite(r["ap"]) for r in group),
                   n_no_positive=sum(r["status"]=="no_positive" for r in group))
        row.update({metric: finite_mean([r[metric] for r in group]) for metric in METRICS})
        topology.append(row)
    summary = []
    groups = grouped(topology,("stage","method","target"))
    for key, group in groups.items():
        for variant in ("all","pair_only","mixed_a","mixed_b"):
            subset = group if variant=="all" else [r for r in group if r["variant"]==variant]
            result = dict(zip(("stage","method","target"),key),variant=variant)
            result["metrics"] = {}
            for metric in METRICS:
                # Match AP's reference density to the topologies with positives.
                eligible = [r for r in subset if np.isfinite(r["ap"])] if metric == "density" else subset
                result["metrics"][metric] = family_ci([r[metric] for r in eligible],
                    [r["family_id"] for r in eligible], samples=samples)
            result["density_policy"] = "AP-eligible topologies only; raw densities retained per topology"
            summary.append(result)
    return topology, summary


def paired_ap_differences(topology, samples):
    lookup={(r["stage"],r["method"],r["target"],r["topology_id"]):r for r in topology}
    rows=[]
    for target in TARGETS:
        for method in ("fused_logit","fused_attention"):
            for reference_stage,reference_method in (("untrained",method),("input-baseline","PLV"),
                                                       ("input-baseline","abs_pearson_delta_phase"),
                                                       ("input-baseline","random")):
                values,families=[],[]
                for row in topology:
                    if (row["stage"],row["method"],row["target"]) != ("forecast-trained",method,target):
                        continue
                    ref=lookup.get((reference_stage,reference_method,target,row["topology_id"]))
                    if ref is not None:
                        values.append(row["ap"]-ref["ap"]);families.append(row["family_id"])
                rows.append(dict(target=target,method=method,reference_stage=reference_stage,
                    reference_method=reference_method,ap_difference=family_ci(values,families,samples=samples)))
    return rows


def paired_controls(cache, samples):
    rows=[]
    for (stage,family,variant,index), entry in cache.items():
        if variant != "mixed_a":
            continue
        other=cache[(stage,family,"mixed_b",index)]
        for method in ("fused_logit","fused_attention"):
            row=dict(stage=stage,family_id=family,trajectory_index=index,method=method,
                     comparison="mixed_a_vs_mixed_b_same_initial_condition",
                     **matrix_comparison(entry[method],other[method]))
            row.update(triad_projection_change=matrix_comparison(entry["a3"],other["a3"])["mean_abs_difference"],
                       pair_skeleton_change=matrix_comparison(entry["a2"],other["a2"])["mean_abs_difference"])
            rows.append(row)
    # Stability across independent ICs: canonicalize node indices before comparing.
    for (stage,family,variant,index), entry in cache.items():
        if index!=0:
            continue
        others=sorted(k for k in cache if k[:3]==(stage,family,variant) and k[3]>0)
        for key in others:
            other=cache[key]
            for method in ("fused_logit","fused_attention"):
                rows.append(dict(stage=stage,family_id=family,trajectory_index=key[3],method=method,
                    comparison=f"{variant}_independent_initial_conditions",
                    **matrix_comparison(entry[method],other[method]),triad_projection_change=0.,pair_skeleton_change=0.))
    summary=[]
    for key,group in grouped(rows,("stage","method","comparison")).items():
        # Collapse repeated ICs first; family remains the unit of resampling.
        fam=grouped(group,("family_id",))
        row=dict(zip(("stage","method","comparison"),key))
        for metric in ("mean_abs_difference","spearman","triad_projection_change"):
            row[metric]=family_ci([finite_mean([r[metric] for r in g]) for g in fam.values()],
                                 [k[0] for k in fam],samples=samples)
        summary.append(row)
    return rows,summary


def forecast_summary(run_dir, samples):
    path=run_dir/"metrics_forecast.json"
    if not path.exists():
        return None
    metrics=json.loads(path.read_text())
    metadata=metrics["test_trajectory_order"]
    result=dict(parameter_counts=metrics["parameter_counts"], models={},differences=[])
    per_topology={}
    for name,m in metrics["models"].items():
        values=[]
        for i,meta in enumerate(metadata):
            values.append(dict(**meta,mse=m["mse_per_trajectory"][i],
                               circular_mae=m["circular_mae_per_trajectory"][i]))
        topo=[]
        for key,g in grouped(values,("topology_id","family_id","variant")).items():
            topo.append(dict(zip(("topology_id","family_id","variant"),key),
                mse=finite_mean([r["mse"] for r in g]),circular_mae=finite_mean([r["circular_mae"] for r in g])))
        per_topology[name]=topo
        result["models"][name]={metric:family_ci([r[metric] for r in topo],
            [r["family_id"] for r in topo],samples=samples) for metric in ("mse","circular_mae")}
        result["models"][name]["by_variant"]={v:{metric:family_ci([r[metric] for r in topo if r["variant"]==v],
            [r["family_id"] for r in topo if r["variant"]==v],samples=samples)
            for metric in ("mse","circular_mae")} for v in ("pair_only","mixed_a","mixed_b")}
    for baseline in ("node-only","persistence"):
        reference={r["topology_id"]:r for r in per_topology[baseline]}
        full=per_topology["forecast-trained"]
        result["differences"].append(dict(comparison=f"forecast-trained minus {baseline}",
            mse=family_ci([r["mse"]-reference[r["topology_id"]]["mse"] for r in full],
                          [r["family_id"] for r in full],samples=samples)))
    result["per_topology"]=per_topology
    return result


def plot_figures(out, example, paired, summary, run_dir, identity):
    figures=fresh_directory(out/"figures")
    n=example["x_obs"].shape[0]
    # This example is fixed before metric computation: first test family, mixed_a, IC 0.
    for stage in ("untrained","forecast-trained"):
        if stage not in example:
            continue
        data=example[stage]
        a=data["attention_weights"]
        fig,axes=plt.subplots(2,4,figsize=(13,6),layout="constrained")
        vmax=float(np.quantile(a,.995))
        for l in range(2):
            for h in range(4):
                im=axes[l,h].imshow(a[l,h],vmin=0,vmax=vmax,cmap="viridis")
                axes[l,h].set_title(f"L{l+1} H{h+1}")
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7)
        fig.suptitle(f"{stage} | head | attention | pairwise / projected-triad probe\n{example['trajectory_id']} | run {identity}",fontsize=10)
        name="attention_heads_example.png" if stage=="forecast-trained" else "attention_heads_untrained.png"
        fig.savefig(figures/name,dpi=150);plt.close(fig)
    stage="forecast-trained" if "forecast-trained" in example else "untrained"
    data=example[stage]
    matrices=[data["fused_logit"], example["a2"],example["a3"],np.maximum(example["a2"],example["a3"])]
    titles=["fused logits (z-score mean)","pairwise truth","projected-triad truth","any-interaction projection"]
    fig,axes=plt.subplots(1,4,figsize=(15,4),layout="constrained")
    for ax,matrix,title in zip(axes,matrices,titles):
        im=ax.imshow(matrix,cmap="coolwarm" if title==titles[0] else "Greys")
        ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle(f"{stage} | fused | logits | truth only for evaluation\n{example['trajectory_id']} | run {identity}",fontsize=10)
    fig.savefig(figures/"fused_vs_ground_truth.png",dpi=150);plt.close(fig)
    methods=[f"L{l}H{h}_{kind}" for l in (1,2) for h in range(1,5) for kind in ("logit_sym","attn_sym")]
    methods += ["fused_logit","fused_attention","PLV","abs_pearson_delta_phase","random"]
    fig,axes=plt.subplots(2,1,figsize=(16,9),layout="constrained")
    for ax,target in zip(axes,("pairwise","projected-triad")):
        for s,offset,color in (("untrained",-.18,"#8899aa"),("forecast-trained",.18,"#286bb2")):
            vals=[]
            for method in methods:
                st="input-baseline" if method in methods[-3:] else s
                entry=next((r for r in summary if (r["stage"],r["method"],r["target"],r["variant"])==
                            (st,method,target,"all")),None)
                vals.append(entry["metrics"]["ap"]["mean"] if entry else np.nan)
            ax.bar(np.arange(len(methods))+offset,vals,width=.35,label=s,color=color)
        ax.set_xticks(range(len(methods)),methods,rotation=60,ha="right",fontsize=7)
        ax.set_ylabel("AP (topology mean)");ax.set_title(target);ax.legend()
    fig.suptitle(f"head / fused | logits / attention / input baselines | run {identity}\nprojected-triad excludes no-positive pair_only; smoke is pipeline validation",fontsize=10)
    fig.savefig(figures/"head_ap_comparison.png",dpi=150);plt.close(fig)
    upper=np.triu_indices(n,1)
    y=example["a2"][upper].astype(bool)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,method in zip(axes,("fused_logit","fused_attention")):
        v=data[method][upper]
        bins=np.linspace(v.min(),v.max()+1e-10,20)
        ax.hist(v[~y],bins=bins,density=True,alpha=.6,label="pairwise negative")
        ax.hist(v[y],bins=bins,density=True,alpha=.6,label="pairwise positive")
        ax.set_title(method);ax.legend(fontsize=8)
    fig.suptitle(f"{stage} | fused | logits / attention | pairwise\nfixed example | run {identity}",fontsize=10)
    fig.savefig(figures/"score_distributions.png",dpi=150);plt.close(fig)
    histories=[]
    import csv
    for kind,path in (("forecast-trained",run_dir/"train_history.csv"),("node-only",run_dir/"node_only/train_history.csv")):
        if path.exists():
            with open(path) as f:
                histories.append((kind,list(csv.DictReader(f))))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for kind,history in histories:
        for field,style in (("train_mse","--"),("val_mse","-")):
            axes[0].plot([int(r["epoch"]) for r in history],[float(r[field]) for r in history],style,label=f"{kind} {field}")
    forecast_path=run_dir/"metrics_forecast.json"
    if forecast_path.exists():
        forecast=json.loads(forecast_path.read_text())
        axes[0].axhline(forecast["validation_persistence"]["mse"],color="gray",linestyle=":",label="val persistence")
        for name,metrics in forecast["models"].items():
            axes[1].plot(np.arange(1,len(metrics["mse_per_horizon"])+1),metrics["mse_per_horizon"],label=name)
    axes[0].set_xlabel("epoch");axes[0].set_ylabel("forecast MSE")
    axes[1].set_xlabel("future step");axes[1].set_ylabel("test forecast MSE")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f"Forecast objective only; structural labels excluded | run {identity}",fontsize=10)
    fig.savefig(figures/"forecast_learning_curves.png",dpi=150);plt.close(fig)
    if paired:
        a,b=paired
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
        for ax,method in zip(axes[:2],("fused_logit","fused_attention")):
            im=ax.imshow(np.abs(a[method]-b[method]),cmap="magma")
            ax.set_title(f"|mixed_a - mixed_b|\n{method}",fontsize=10);fig.colorbar(im,ax=ax,shrink=.7)
        im=axes[2].imshow(np.abs(a["a3"].astype(float)-b["a3"].astype(float)),vmin=0,vmax=1,cmap="Greys")
        axes[2].set_title("projected-triad truth change",fontsize=10);fig.colorbar(im,ax=axes[2],shrink=.7)
        fig.suptitle(f"{stage} | fused | logits / attention | pairwise skeleton fixed\nsame omega / initial condition; changes do not establish triad identification | run {identity}",fontsize=9)
        fig.savefig(figures/"paired_topology_control.png",dpi=150);plt.close(fig)


def run_probe(config, data, checkpoint, out, device="cpu", stage_choice="both"):
    device=get_device(device)
    seed_all(config["training"]["seed"],config["training"]["num_threads"])
    root=Path(data); checkpoint=Path(checkpoint)
    manifest=json.loads((root/"manifest.json").read_text())
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    if ckpt["node_only"]:
        raise ValueError("A node-only checkpoint has no attention")
    if ckpt["data_manifest_sha256"]!=sha256_file(root/"manifest.json"):
        raise ValueError("Checkpoint and data manifest differ")
    for key in ("data","model"):
        if config[key]!=ckpt["config"][key]:
            raise ValueError(f"Configuration mismatch: {key}")
    if stage_choice in ("both","trained") and ckpt["epoch"]==0:
        raise ValueError("Use --stage untrained for a checkpoint with epoch=0")
    out=fresh_directory(out); identity=run_id(out)
    write_json(out/"environment.json",dict(run_id=identity,**environment()))
    matrices_dir=fresh_directory(out/"matrices")
    indices=np.load(root/"candidates.npz")
    pairs,tris=indices["pair_index"],indices["tri_index"]
    n=config["data"]["n_nodes"]; upper=np.triu_indices(n,1)
    observed=config["data"]["observed"]
    stages=["untrained","forecast-trained"] if stage_choice=="both" else ["untrained" if stage_choice=="untrained" else "forecast-trained"]
    models={}
    for stage in stages:
        model=AttentionProbe(ckpt["config"]).to(device).eval()
        model.load_state_dict(ckpt["initial_state"] if stage=="untrained" else ckpt["model_state"])
        models[stage]=model
    rows=[]; cache={}; examples=[]; fixed_example=None
    test=[t for t in manifest["topologies"] if t["split"]=="test"]
    first_family=test[0]["family_id"]
    with torch.inference_mode():
        for topo in test:
            arrays=dict(np.load(root/topo["file"]))
            inputs=arrays["x_all"][:,:,:observed]
            # The model only sees historical observations; all truth is processed below.
            outputs={stage:{k:v.cpu().numpy() for k,v in model(torch.from_numpy(inputs).to(device)).items()
                            if k in ("attention_logits","attention_weights")}
                     for stage,model in models.items()}
            export=dict(run_id=identity, data_run_id=manifest["run_id"], family_id=topo["family_id"],
                        topology_id=topo["topology_id"], variant=topo["variant"], permutation=arrays["permutation"])
            stage_scores={stage:[] for stage in stages}
            for idx,x in enumerate(inputs):
                a2,a3=label_matrices(arrays["y_pair"][idx],arrays["y_tri"][idx],pairs,tris,n)
                any_interaction=np.maximum(a2,a3)
                truth={"pairwise":a2[upper],"projected-triad":a3[upper],
                       "any-interaction":any_interaction[upper],"triad-only-subset":a3[upper]}
                masks={target:np.ones(len(pairs),dtype=bool) for target in TARGETS}
                masks["triad-only-subset"]=~a2[upper].astype(bool)
                baseline=association_baselines(x,ckpt["seed"]+100*topo["family_id"]+idx)
                measurements={"input-baseline":baseline}
                for stage,output in outputs.items():
                    scores=score_matrices(output["attention_logits"][idx],output["attention_weights"][idx])
                    stage_scores[stage].append(scores)
                    measurements[stage]=scores
                    inverse=np.argsort(arrays["permutation"][idx])
                    canonical=lambda m:m[inverse][:,inverse]
                    cache[(stage,topo["family_id"],topo["variant"],idx)]={
                        **{method:canonical(scores[method]) for method in ("fused_logit","fused_attention")},
                        "a2":canonical(a2),"a3":canonical(a3)}
                meta={k:topo[k] for k in ("family_id","variant","topology_id")}
                trajectory_id=f"{topo['topology_id']}_t{idx:03d}"
                for stage,scores in measurements.items():
                    for method,matrix in scores.items():
                        for target in TARGETS:
                            mask=masks[target]
                            rows.append(dict(run_id=identity,**meta,trajectory_id=trajectory_id,
                                stage=stage,method=method,target=target,
                                **binary_metrics(truth[target][mask],matrix[upper][mask])))
                if topo["family_id"]==first_family and idx==0:
                    ex=dict(x_obs=x,y_pair=arrays["y_pair"][idx],y_tri=arrays["y_tri"][idx],
                        family_id=topo["family_id"],variant=topo["variant"],trajectory_id=trajectory_id,a2=a2,a3=a3)
                    for stage,output in outputs.items():
                        ex[stage]=dict(attention_logits=output["attention_logits"][idx],
                            attention_weights=output["attention_weights"][idx],**stage_scores[stage][-1])
                    examples.append(ex)
                    if topo["variant"]=="mixed_a":
                        fixed_example=ex
            for stage,output in outputs.items():
                for name,value in output.items():
                    export[f"{stage}_{name}"]=value
                methods=list(stage_scores[stage][0])
                export[f"{stage}_score_names"]=np.array(methods)
                export[f"{stage}_scores"]=np.stack([[sc[m] for m in methods] for sc in stage_scores[stage]])
            np.savez_compressed(matrices_dir/(topo["topology_id"]+".npz"),**export)
            print(f"probed {topo['topology_id']} trajectories={len(inputs)}",flush=True)
    samples=config["evaluation"]["bootstrap_samples"]
    topology,summary=aggregate(rows,samples)
    paired_rows,paired_summary=paired_controls(cache,samples)
    for row in paired_rows:
        row["run_id"]=identity
    write_csv(out/"metrics_per_head.csv",rows)
    write_csv(out/"metrics_per_topology.csv",topology)
    write_csv(out/"paired_controls.csv",paired_rows)
    example_count=config["evaluation"]["example_count"]
    examples=examples[:example_count]
    archive=dict(run_id=identity,selection="first test family, IC 0, variants in manifest order; no AP selection",
                 **{name:np.array([e[name] for e in examples]) for name in
                    ("x_obs","y_pair","y_tri","family_id","variant","trajectory_id")})
    for stage in stages:
        for name in ("attention_logits","attention_weights","fused_logit","fused_attention"):
            key=name if stage==stages[-1] else f"untrained_{name}"
            archive[key]=np.stack([e[stage][name] for e in examples])
        for kind,raw in (("R_logit","attention_logits"),("R_attn","attention_weights")):
            key=kind if stage==stages[-1] else f"untrained_{kind}"
            archive[key]=np.stack([symmetric(e[stage][raw]) for e in examples])
    np.savez_compressed(out/"attention_examples.npz",**archive)
    results=dict(run_id=identity,data_run_id=manifest["run_id"],training_run_id=ckpt["run_id"],
        experiment=config["name"],stages=stages,model_seed=ckpt["seed"],checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint),checkpoint_epoch=ckpt["epoch"],
        n_test_families=len({t["family_id"] for t in test}),n_test_topologies=len(test),
        n_test_trajectories=len(test)*config["data"]["trajectories"],
        aggregation="trajectory -> topology mean; cluster bootstrap whole families, 1000 draws unless configured",
        no_positive_policy="AP/Precision@K/AUROC null (CSV NaN); all-positive AUROC null; no filtering by model performance",
        summary=summary,paired_ap_differences=paired_ap_differences(topology,samples),
        paired_controls=paired_summary,forecast=forecast_summary(checkpoint.parent,samples),
        example_selection=archive["selection"],
        model_seed_stability="Not assessed unless separate seeds are run; IC stability is not model-seed stability")
    write_json(out/"metrics_summary.json",results)
    write_json(out/"baselines.json",dict(run_id=identity,
        summary=[r for r in summary if r["stage"]=="input-baseline"],
        definition="Input history only; random symmetric upper triangle uniform, seed=model_seed+100*family+IC",
        forecast=results["forecast"]))
    st=stages[-1]
    pair=(cache[(st,first_family,"mixed_a",0)],cache[(st,first_family,"mixed_b",0)])
    plot_figures(out,fixed_example,pair,summary,checkpoint.parent,identity)
    print(f"PROBE_PATH={out.resolve()} RUN_ID={identity}",flush=True)
    return results


def main():
    parser=argparse.ArgumentParser()
    for name in ("config","data","checkpoint","out"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--stage",choices=("both","untrained","trained"),default="both")
    args=parser.parse_args()
    run_probe(read_config(args.config),args.data,args.checkpoint,args.out,args.device,args.stage)


if __name__=="__main__":
    main()
