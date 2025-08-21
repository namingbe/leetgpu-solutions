#!/usr/bin/env python3

import requests
from pathlib import Path

challenges = requests.get("https://api.leetgpu.com/api/v1/challenges/fetch-all").json()

for challenge in challenges:
    title = challenge['title'].lower().replace(' ', '-')
    dir_path = Path(f"challenges/{title}")
    if dir_path.exists():
        continue
    starter = requests.get(f"https://api.leetgpu.com/api/v1/challenges/{challenge['id']}/starter-code").json()
    dir_path.mkdir(parents=True)
    for lang, code in starter['starter_code'].items():
        if lang == 'pytorch':
            lang = 'torch'
        ext = 'cu' if lang == 'cuda' else 'py'
        (dir_path / f"{lang}.{ext}").write_text(code)
        print(f"Saved {title}/{lang}.{ext}")
