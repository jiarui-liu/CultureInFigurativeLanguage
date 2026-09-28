#!/bin/bash
# Drives the 2B bidirectional pipeline: submits each step once its inputs exist.
# Run on the login/vscode node:  nohup bash orchestrate.sh > $B/logs/orchestrate.log 2>&1 &
# State (what was submitted) lives in $B/state/ as marker files.
BIDIR=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
ST=$B/state; mkdir -p $ST
LANGS=${LANGS:-"ar hi zh"}
SEEDS=${SEEDS:-"42"}
# main models = the post-trained Qwen3.5-2B (same model type as the 9B study, which used
# Qwen/Qwen3.5-9B); Qwen3.5-2B-Base runs are kept as an ablation (run prefix "" vs "i_")
MODELS=${MODELS:-"i_:Qwen/Qwen3.5-2B :Qwen/Qwen3.5-2B-Base"}
PRE="--partition=preempt --qos=preempt_qos --requeue"
TASKS_ar="kinayat_meaning+kinayat_cloze+ar_figurative+alyah+dzirieval+arabculture+arabic_cultural_qa+global_piqa_ar+global_piqa_ar_parallel+arabicmmlu+jsonl:idiomatlas_mc_ar_seen+jsonl:idiomatlas_mc_ar_unseen+jsonl:symbolism_ar"
TASKS_hi="mabl+global_piqa+milu+jsonl:idiomatlas_mc_hi_seen+jsonl:idiomatlas_mc_hi_unseen+jsonl:symbolism_hi"
TASKS_zh="chengyu_bench+chengyu_bench_app+chid+ccpm+cmmlu+jsonl:idiomatlas_mc_zh_seen+jsonl:idiomatlas_mc_zh_unseen+jsonl:symbolism_zh"
once() { [ -f $ST/$1 ] && return 1; touch $ST/$1; return 0; }
log() { echo "$(date '+%m-%d %H:%M') $*"; }
cd $BIDIR
while true; do
  for l in $LANGS; do
    # 1) classifier + scoring
    if [ -s $B/annot/$l.jsonl ] && once clf_$l; then
      log "submit classify $l"; sbatch --array=0-7 --export=ALL,LANG_=$l,NPARTS=8 classify.slurm
    fi
    np=$(ls $B/pool/$l/*.jsonl.gz 2>/dev/null | wc -l); ns=$(ls $B/scores/$l/*.scores.jsonl 2>/dev/null | wc -l)
    # 2) culture arm
    if [ "$np" -gt 0 ] && [ "$ns" -eq "$np" ] && [ -f $B/arms/$l/.base_done ] && once culture_$l; then
      log "prep culture $l"; nohup bash prep_arms_local.sh $l culture 2 > $B/logs/preplocal-$l-culture.log 2>&1 &
    fi
    # 3) notes
    if [ -f $B/arms/$l/.culture_done ] && once notes_$l; then
      log "submit notes $l"; sbatch --array=0-3 --export=ALL,LANG_=$l,NPARTS=4 notes.slurm
    fi
    nn=$(ls $B/arms/$l/notes/notes_*.jsonl.gz 2>/dev/null | wc -l)
    if [ "$nn" -eq 4 ] && once notesarm_$l; then
      log "prep notes_arm $l"; nohup bash prep_arms_local.sh $l notes_arm 2 > $B/logs/preplocal-$l-notes.log 2>&1 &
    fi
    # 4) training + evaluation per packed arm, model and seed (post-trained model first)
    for ms in $MODELS; do
      pre=${ms%%:*}; MODEL=${ms#*:}
      for arm in random idiom_tagged idiom_untagged culture culture_notes; do
        [ -f $B/packed/$l/$arm.json ] || continue
        for s in $SEEDS; do
          run=${pre}$arm; [ $s != 42 ] && run=${run}_s$s
          if once train_${l}_$run; then
            log "submit train $l $run ($MODEL)"; sbatch $PRE --job-name=t-$l-$run --export=ALL,MODEL=$MODEL,DATA=$B/packed/$l/$arm,OUT=$B/ckpt/$l/$run,SEED=$s train_cpt.slurm
          fi
          if [ -f $B/ckpt/$l/$run/train_manifest.json ] && once eval_${l}_$run; then
            T=TASKS_$l; log "submit eval $l $run"; sbatch $PRE --export=ALL,MODEL=$B/ckpt/$l/$run,OUT=$B/eval2b/$l/$run,TASKS=${!T},BS=4 eval.slurm
          fi
        done
      done
      if once eval_${l}_${pre}base; then
        T=TASKS_$l; log "submit eval $l ${pre}base"; sbatch $PRE --export=ALL,MODEL=$MODEL,OUT=$B/eval2b/$l/${pre}base,TASKS=${!T},BS=4 eval.slurm
      fi
    done
  done
  sleep 120
done
