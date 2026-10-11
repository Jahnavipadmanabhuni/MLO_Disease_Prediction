#!/usr/bin/env bash
# End-to-end demo of the whole loop: train -> serve -> monitor -> drift -> retrain -> recover.
set -e
python -m src.train
echo; echo "##### 1. Healthy traffic -> expect no drift"
python -m src.simulate_traffic --n 300 --drift 0.0 --reset
python -m src.monitor | python -c "import sys,json;t=sys.stdin.read();s=json.loads(t[t.index('{'):]);print(s['status'],s['reasons'])"
echo; echo "##### 2. Patient patterns change (outbreak + new COVID variant) -> expect drift"
python -m src.simulate_traffic --n 400 --drift 1.0 --reset
python -m src.monitor | python -c "import sys,json;t=sys.stdin.read();s=json.loads(t[t.index('{'):]);print(s['status'],s['reasons'])"
echo; echo "##### 3. Controlled retraining"
python -m src.retrain | tail -25
echo; echo "##### 4. Same shifted world, new champion -> expect recovery"
python -m src.simulate_traffic --n 400 --drift 1.0 --seed 777 --reset
python -m src.monitor | python -c "import sys,json;t=sys.stdin.read();s=json.loads(t[t.index('{'):]);print(s['status'],s['reasons'],s['performance'])"
