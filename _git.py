# -*- coding: utf-8 -*-
import subprocess
cwd = r'd:\Trae的工作空间\刺痛说'
for cmd in [['git','add','-A'],
            ['git','commit','-m','缠论结构推演精细化+创业板指缠论分析'],
            ['git','push']]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    label = ' '.join(cmd)
    print(f'>>> {label}')
    print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    print(f'RC={r.returncode}')
