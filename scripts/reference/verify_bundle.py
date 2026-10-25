"""Verify every file in the published reference-data manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'benchmarks/score_reference/manifest.json').read_text())
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path escapes dataset root')
        payload = path.read_bytes()
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {entry["path"]}')
    print(f'Verified {len(manifest["files"])} files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/3DKoreanMelon'))
    verify(parser.parse_args().dataset)
