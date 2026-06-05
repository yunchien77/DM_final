#!/usr/bin/env bash
# Wait for the baseline pipeline to finish, then run the enhanced (team_feat) pipeline.
echo "[chain] $(date) waiting for baseline (run_team.sh ... team_base) to exit ..."
sleep 20
while pgrep -f "run_team.sh /mnt/1stHDD/juiyun/DMFP/team_base" >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] $(date) baseline process gone -> launching enhanced (team_feat)"
bash /mnt/1stHDD/juiyun/DMFP/run_team.sh \
    /mnt/1stHDD/juiyun/DMFP/team_feat \
    /mnt/1stHDD/juiyun/DMFP/submission_teamv2_feat.csv \
    feat
echo "[chain] $(date) enhanced finished"
