import json
d=json.load(open('artifacts/test_metrics.json'))
r=d['report']
for k,v in r.items():
    if isinstance(v,dict) and 'precision' in v:
        print(f"{k:12} {v['precision']:.3f} {v['recall']:.3f} {v['f1-score']:.3f} {v['support']}")
print('---')
print('acc', d['test_acc'])
