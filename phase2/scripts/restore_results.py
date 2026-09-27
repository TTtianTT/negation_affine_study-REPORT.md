"""Restore losslessly archived results without overwriting differing files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'results'
    output = args.output_dir or source
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
    for entry in json.loads((source / 'ARCHIVES.json').read_text()):
        archive = source / entry['archive']
        if digest(archive) != entry['archive_sha256']:
            raise ValueError(f'Archive checksum mismatch: {archive}')
        target = output / entry['file']
        h = hashlib.sha256()
        size = 0
        with tempfile.TemporaryFile() as tmp, gzip.open(archive, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
                size += len(block)
                if not args.verify_only:
                    tmp.write(block)
            if size != entry['bytes'] or h.hexdigest() != entry['sha256']:
                raise ValueError(f'Original checksum mismatch: {archive}')
            if not args.verify_only:
                if target.exists():
                    if digest(target) != entry['sha256']:
                        raise FileExistsError(f'Refusing to overwrite: {target}')
                else:
                    tmp.seek(0)
                    with target.open('xb') as dest:
                        for block in iter(lambda: tmp.read(1024 * 1024), b''):
                            dest.write(block)
        print(f'Verified {entry["file"]}: {size} bytes')


if __name__ == '__main__':
    main()
