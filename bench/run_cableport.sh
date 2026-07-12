#!/bin/bash
# Cable-port validation driver: coaxs x solver-configs, each case in its own process,
# then aggregate + PASS/FAIL verdict. Run inside the bench container (complex mode):
#   docker run -d -v <tree>:/work -w /work/bench/cableport scatt3d-bench bash /work/hunter-Scatt3D/bench/run_cableport.sh
# Env overrides: CP_COAXS="0 3 5" CP_CONFIGS='lu={} sym={"symmetric": true}' CP_H CP_DEG CP_NF
set -u
source /usr/local/bin/dolfinx-complex-mode
export PYTHONUNBUFFERED=1
export SCATT3D_SRC=${SCATT3D_SRC:-/work/hunter-Scatt3D/Scatt3D}
BENCH=$(dirname "$0")
CP_COAXS=${CP_COAXS:-"0 3 5"}
CP_H=${CP_H:-0.2857142857142857}   # 1/3.5
CP_DEG=${CP_DEG:-3}
CP_NF=${CP_NF:-3}
# solver configs: parallel arrays, extend via CP_EXTRA_NAME/CP_EXTRA_JSON
CFG_NAMES=(lu sym)
CFG_JSONS=('{}' '{"symmetric": true}')
if [ -n "${CP_EXTRA_NAME:-}" ]; then CFG_NAMES+=("$CP_EXTRA_NAME"); CFG_JSONS+=("$CP_EXTRA_JSON"); fi

: > results.jsonl
: > exits.txt
for c in $CP_COAXS; do
  for i in "${!CFG_NAMES[@]}"; do
    name=${CFG_NAMES[$i]}; js=${CFG_JSONS[$i]}
    log="case_c${c}_${name}.log"
    echo "=== coax $c config $name ($js) ===" | tee -a driver.log
    python3 "$BENCH/cableport_validate.py" "$c" "$js" "$name" "$CP_H" "$CP_DEG" "$CP_NF" > "$log" 2>&1
    ec=$?
    echo "COAX $c CFG $name EXIT $ec" >> exits.txt
    grep -h '^CABLEPORT_RESULT ' "$log" | sed 's/^CABLEPORT_RESULT //' >> results.jsonl
  done
done

python3 - <<'PYEOF'
import json, math
res = {}
for line in open('results.jsonl'):
    r = json.loads(line)
    res[(r['coax'], r['tag'])] = r
exits = {}
for line in open('exits.txt'):
    p = line.split()
    exits[(int(p[1]), p[3])] = int(p[5])
cfgs = sorted({t for _, t in exits}, key=lambda t: t != 'lu')
coaxs = sorted({c for c, _ in exits})
def s_arr(r): return [complex(a, b) for a, b in r['S_sim']]
print(f"\n{'coax':>4} {'cfg':>6} {'exit':>4} {'max_mag_relerr':>15} {'max_phase_deg':>13} {'max|dS| vs lu':>13} {'factMB(max/tot)':>16}")
overall = True
for c in coaxs:
    lu = res.get((c, 'lu'))
    for t in cfgs:
        ec = exits.get((c, t), -1)
        r = res.get((c, t))
        if ec != 0 or r is None:
            print(f"{c:>4} {t:>6} {ec:>4} {'CRASH/NO-RESULT':>15}")
            overall = False
            continue
        dS = max(abs(a - b) for a, b in zip(s_arr(r), s_arr(lu))) if (lu and t != 'lu') else 0.0
        fm = r.get('factor_mem_mb') or ['-', '-']
        print(f"{c:>4} {t:>6} {ec:>4} {max(r['mag_relerr']):>15.3e} {max(r['phase_err_deg']):>13.3e} {dS:>13.3e} {str(fm):>16}")
        if t != 'lu':
            tol = 1e-6 if 'blr' in json.dumps(r['solver']) else 1e-9
            if dS > tol:
                print(f"     ^ FAIL: dS {dS:.3e} > {tol}")
                overall = False
print("\nVERDICT:", "PASS" if overall else "FAIL")
PYEOF
