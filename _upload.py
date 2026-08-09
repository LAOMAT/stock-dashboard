# -*- coding: utf-8 -*-
"""初始化git仓库 + 创建GitHub仓库 + 推送"""
import subprocess, os

cwd = r'd:\Trae的工作空间\刺痛说'

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, **kw)
    print(f'>>> {" ".join(cmd)}')
    if r.stdout: print(r.stdout.strip())
    if r.stderr: print(r.stderr.strip())
    print(f'RC={r.returncode}')
    return r

# 1. git init + add + commit
run(['git', 'init'])
run(['git', 'add', '-A'])
run(['git', 'commit', '-m', '市场趋势监测看板V3: 缠论/量柱/因果/回测自进化+走势推演+MRS周期拐点'])

# 2. 创建GitHub仓库(私有)
r = run(['gh', 'repo', 'create', 'stock-dashboard', '--public', '--source=.', '--push',
         '--description', 'A股市场趋势监测看板: 缠论+量柱+波浪+因果推理+回测自进化'])
if r.returncode != 0:
    # 如果public失败,尝试private
    print('public创建失败,尝试private...')
    run(['gh', 'repo', 'create', 'stock-dashboard', '--private', '--source=.', '--push',
         '--description', 'A股市场趋势监测看板: 缠论+量柱+波浪+因果推理+回测自进化'])

# 3. 获取仓库URL
r = run(['gh', 'repo', 'view', '--json', 'url'])
print('\n=== 仓库地址 ===')
print(r.stdout)
