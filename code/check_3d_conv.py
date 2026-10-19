import json, glob, os, sys
for root in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(root), '**', 'meta.json'), recursive=True)):
        m = json.load(open(p))
        if not p.endswith('meta.json') or 'march' not in m:
            continue
        b = m['march']['bouss']; c = m['march']['comp']
        print('%-52s eps=%.2f Ra=%-7.0f b=%.1e c=%.1e ok=%s/%s' % (
            p.split('/nc_run/')[-1].replace('/meta.json', ''), m['dT'], m['Ra'],
            b.get('rel_change_final', -1), c.get('rel_change_final', -1), b.get('ok'), c.get('ok')))
