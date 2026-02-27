#!/bin/bash
# GLMB test: exact reproduction of history_004 scenario
# TARGET and SENSOR parameters are FIXED - only GLMB algorithm changes
docker exec fasttracker-fasttracker-1 /app/fasttracker \
  --scenario single-ballistic \
  --missile-type ballistic \
  --num-targets 1 \
  --duration 645.4 \
  --framerate 1.0 \
  --launch-x -905389.9 --launch-y 575544.9 \
  --target-x 161.6 --target-y -109.3 \
  --launch-angle 1.0821041362364843 \
  --boost-duration 65.0 --boost-accel 30.0 \
  --initial-mass 20000.0 --fuel-fraction 0.78 \
  --specific-impulse 250.0 --drag-coefficient 0.3 --cross-section 1.0 \
  --sensor-x -773329.3 --sensor-y -98854.2 \
  --radar-min-range 0.0 --radar-max-range 0.0 \
  --azimuth-coverage 2.0944 --min-elevation -0.5236 --max-elevation 1.5708 \
  --enable-separation --warhead-mass-fraction 0.3 \
  --pfa 1e-05 --pd-ref 0.75 --pd-ref-range 600000.0 \
  --range-noise 97.0 --azimuth-noise 0.004363323129985824 \
  --elevation-noise 0.008726646259971648 --doppler-noise 22.0 \
  --gate-threshold 10.0 --confirm-hits 4 --confirm-window 5 --delete-misses 10 \
  --min-snr 10.0 \
  --process-pos-noise 300.0 --process-vel-noise 150.0 --process-acc-noise 300.0 \
  --ukf-alpha 0.5 --ukf-beta 2.0 --ukf-kappa 0.0 \
  --max-distance 100000.0 --max-jump-velocity 10000.0 --min-init-distance 30000.0 \
  --imm-ca-ca 0.8 --imm-ca-bal 0.15 --imm-ca-ct 0.05 \
  --imm-bal-ca 0.1 --imm-bal-bal 0.85 --imm-bal-ct 0.05 \
  --imm-ct-ca 0.05 --imm-ct-bal 0.1 --imm-ct-ct 0.85 \
  --imm-ca-noise 0.1 --imm-bal-noise 0.3 --imm-ct-noise 2.5 \
  --num-runs 10 --seed 1 \
  --cluster-count 1 --cluster-spread 50000.0 --launch-time-spread 300.0 \
  --beam-width 0.05235987755982988 --num-beams 50 --min-search-beams 1 \
  --track-confirmed-only \
  --search-sector 0.5235987755982988 \
  --search-center 1.5882496193148399 \
  --antenna-boresight 0.9599310885968813 \
  --search-min-range 500000.0 --search-max-range 810000.0 \
  --track-range-width 150000.0 --range-resolution 50.0 \
  --elev-scan-min 0.03490658503988659 --elev-scan-max 0.03490658503988659 \
  --elev-bars-per-frame 3 --elev-cycle-steps 9 \
  --association glmb \
  --glmb-pd 0.85 --glmb-k-best 5 --glmb-max-hypotheses 50 \
  --glmb-score-decay 0.9 --glmb-survival 0.99 \
  --glmb-birth-weight 0.01 --glmb-clutter-density 1e-06 \
  --glmb-init-existence 0.2 \
  2>&1 | grep -E "Aggregated|RMSE|OSPA|Precision|Recall|F1|Wall-clock|Avg frame|Detection|Clutter|False Track|Track Purity|Mostly|Total created|Total confirmed"
