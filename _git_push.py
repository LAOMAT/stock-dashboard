# -*- coding: utf-8 -*-
import subprocess
cwd = r'd:\Trae的工作空间\刺痛说'
for cmd in [['git','add','-A'],
            ['git','commit','-m','移动端适配: 响应式布局+图表高度自适应+触摸滑动+热力图横向滚动'],
            ['git','push']]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    label = ' '.join(cmd)
    print(f'>>> {label}')
    print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    print(f'RC={r.returncode}')
