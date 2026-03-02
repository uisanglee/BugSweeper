import csv
import sys
from collections import defaultdict
from statistics import mean, pstdev

path = sys.argv[1] if len(sys.argv) > 1 else 'models/metrics_4_ce.csv'

rows = []
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    print('No rows found.')
    sys.exit(0)

groups = defaultdict(list)
for r in rows:
    key = (r['pool_mode'], r['max_pool_levels'], r['coverage'])
    groups[key].append(float(r['macro_f1']))

print('pool_mode,max_pool_levels,coverage,macro_f1_mean,macro_f1_std,n')
for key in sorted(groups.keys()):
    vals = groups[key]
    std = pstdev(vals) if len(vals) > 1 else 0.0
    print(f'{key[0]},{key[1]},{key[2]},{mean(vals):.4f},{std:.4f},{len(vals)}')
