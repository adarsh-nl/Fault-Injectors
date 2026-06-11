"""
download_griffin.py
-------------------
Download and extract the Griffin dataset (or a subset) from Hugging Face.

Griffin is hosted at `wjh-svm/Griffin` and is 967 GB in total. Each subset is a
folder of per-modality zip files, so you can fetch only what you need (for the
visualisation tools, metadata + LiDAR + a couple of cameras is enough, ~46 GB).

Two ways to use it
------------------
1. As an importable module from a notebook (it lives in the src package):

       from src.download_griffin import (
           list_catalogue, download_subset, SUBSET_META
       )
       cat = list_catalogue()                       # inspect what is available
       download_subset(
           subset='griffin_50scenes_25m',
           files=['vehicle_metadata.zip', 'vehicle_lidar.zip',
                  'vehicle_camera_front.zip', 'drone_camera_bottom.zip'],
           dest='./datasets', extract=True,
       )

2. As a command-line tool (run it directly by path):

       python src/download_griffin.py                       # fully interactive
       python src/download_griffin.py --subset griffin_50scenes_25m --minimal
       python src/download_griffin.py --subset griffin_50scenes_25m --all
       python src/download_griffin.py --list                # just print the catalogue

Config defaults are at the top; if a required choice is missing it will prompt.
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download


# ── Config defaults (used when not provided via args or prompt) ────────────

REPO_ID   = 'wjh-svm/Griffin'
REPO_TYPE = 'dataset'

DEFAULT_SUBSET = 'griffin_50scenes_25m'
DEFAULT_DEST   = './datasets'

# A small, useful subset for the visualisation tools (~46 GB)
MINIMAL_FILES = [
    'vehicle_metadata.zip',     # calib, labels, poses (~11 MB)
    'drone_metadata.zip',       # drone calib, labels (~15 MB)
    'vehicle_lidar.zip',        # LiDAR point clouds (~214 MB)
    'vehicle_camera_front.zip', # front camera (~16 GB)
    'drone_camera_bottom.zip',  # drone bottom camera (~19 GB)
]

SUBSET_META = {
    'griffin_50scenes_25m':     {'altitude': '25 +/- 2 m', 'scenes': 47,  'total_gb': 167},
    'griffin_50scenes_40m':     {'altitude': '40 +/- 2 m', 'scenes': 54,  'total_gb': 190},
    'griffin_50scenes_55m':     {'altitude': '55 +/- 2 m', 'scenes': 50,  'total_gb': 175},
    'griffin_100scenes_random': {'altitude': '20-60 m',    'scenes': 104, 'total_gb': 435},
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt_size(n):
    if n is None:
        return 'unknown'
    if n >= 1e9:
        return f'{n/1e9:.1f} GB'
    if n >= 1e6:
        return f'{n/1e6:.0f} MB'
    return f'{n/1e3:.0f} KB'


def list_catalogue():
    """
    Query Hugging Face and return the file catalogue.

    Returns
    -------
    dict  {subset_name: [(filename, size_bytes), ...]}, sorted by size desc.
    """
    api = HfApi()
    items = api.list_repo_tree(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        path_in_repo='datasets', recursive=True, expand=True,
    )
    catalogue = defaultdict(list)
    for it in items:
        if not hasattr(it, 'size'):
            continue
        parts = it.path.split('/')
        if len(parts) != 3:          # datasets/<subset>/<file>
            continue
        _, subset, fname = parts
        catalogue[subset].append((fname, it.size))
    for subset in catalogue:
        catalogue[subset].sort(key=lambda x: -(x[1] or 0))
    return dict(catalogue)


def print_catalogue(catalogue=None):
    """Pretty-print the catalogue to stdout."""
    if catalogue is None:
        catalogue = list_catalogue()
    print(f'\n{"="*66}')
    print(f'  Griffin dataset catalogue — https://huggingface.co/datasets/{REPO_ID}')
    print(f'{"="*66}')
    for subset, flist in sorted(catalogue.items()):
        meta = SUBSET_META.get(subset, {})
        print(f'\n  {subset}')
        print(f'    altitude: {meta.get("altitude","?")}   '
              f'scenes: {meta.get("scenes","?")}   '
              f'total: ~{meta.get("total_gb","?")} GB')
        for fname, size in flist:
            print(f'      {_fmt_size(size):>9s}   {fname}')
    print()


def download_subset(subset, files='minimal', dest=DEFAULT_DEST,
                    extract=True, catalogue=None):
    """
    Download (and optionally extract) chosen files of a subset.

    Parameters
    ----------
    subset    : str   one of the keys in SUBSET_META.
    files     : 'minimal' | 'all' | list of zip filenames.
    dest      : str   local directory to download into.
    extract   : bool  unzip into <dest>/<subset>/griffin-release/ when True.
    catalogue : dict  optional pre-fetched catalogue (avoids a second query).

    Returns
    -------
    dict  {'downloaded': [paths], 'extract_root': str or None}
    """
    if catalogue is None:
        catalogue = list_catalogue()
    if subset not in catalogue:
        raise ValueError(f'Unknown subset {subset!r}. Available: {list(catalogue)}')

    available = {f for f, _ in catalogue[subset]}
    size_of   = {f: s for f, s in catalogue[subset]}

    if files == 'all':
        chosen = sorted(available)
    elif files == 'minimal':
        chosen = [f for f in MINIMAL_FILES if f in available]
    else:
        chosen = [f for f in files if f in available]
        missing = [f for f in files if f not in available]
        if missing:
            print(f'  [warn] not in subset, skipping: {missing}')

    if not chosen:
        raise ValueError('No valid files selected to download.')

    total = sum(size_of.get(f, 0) for f in chosen)
    print(f'\nDownload plan for {subset}:')
    print('-' * 50)
    for f in chosen:
        print(f'  {_fmt_size(size_of.get(f,0)):>9s}   {f}')
    print('-' * 50)
    print(f'  {_fmt_size(total):>9s}   TOTAL ({len(chosen)} files)\n')

    dest_root = Path(dest) / subset
    dest_root.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for fname in chosen:
        flat = dest_root / fname
        if flat.exists():
            print(f'  [skip] already present: {fname}')
            downloaded.append(flat)
            continue
        print(f'  [get ] {fname}  ({_fmt_size(size_of.get(fname,0))}) ...')
        local = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=f'datasets/{subset}/{fname}',
            local_dir=str(dest_root), local_dir_use_symlinks=False,
        )
        local_p = Path(local)
        if local_p != flat and local_p.exists():
            shutil.move(str(local_p), str(flat))
        downloaded.append(flat)

    extract_root = None
    if extract:
        extract_root = dest_root / 'griffin-release'
        print(f'\nExtracting to {extract_root} ...')
        for zp in downloaded:
            if not str(zp).endswith('.zip'):
                continue
            print(f'  [unzip] {zp.name}')
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(extract_root)
        print('Extraction complete.')

    print(f'\nDone. {len(downloaded)} files in {dest_root}.')
    return {'downloaded': [str(p) for p in downloaded],
            'extract_root': str(extract_root) if extract_root else None}


def locate_sides(dest, subset):
    """
    After extraction, find the vehicle-side and drone-side directories.

    Returns
    -------
    dict  {'dataset_root': str, 'vehicle_side': str, 'drone_side': str or None}
    """
    release = Path(dest) / subset
    veh   = sorted(release.rglob('vehicle-side'))
    drone = sorted(release.rglob('drone-side'))
    if not veh:
        raise FileNotFoundError(f'vehicle-side not found under {release}')
    return {
        'dataset_root': str(Path(veh[0]).parent),
        'vehicle_side': str(veh[0]),
        'drone_side'  : str(drone[0]) if drone else None,
    }


# ── Interactive prompts (used when a choice is missing) ────────────────────

def _prompt_subset(catalogue):
    subsets = sorted(catalogue)
    print('\nAvailable subsets:')
    for i, s in enumerate(subsets):
        meta = SUBSET_META.get(s, {})
        print(f'  [{i}] {s}  (altitude {meta.get("altitude","?")}, '
              f'~{meta.get("total_gb","?")} GB)')
    while True:
        raw = input(f'Pick a subset [0-{len(subsets)-1}] '
                    f'(Enter = {DEFAULT_SUBSET}): ').strip()
        if raw == '':
            return DEFAULT_SUBSET
        if raw.isdigit() and 0 <= int(raw) < len(subsets):
            return subsets[int(raw)]
        if raw in catalogue:
            return raw
        print('  Invalid choice, try again.')


def _prompt_files(catalogue, subset):
    flist = catalogue[subset]
    print(f'\nFiles in {subset}:')
    for i, (fname, size) in enumerate(flist):
        print(f'  [{i:2d}] {_fmt_size(size):>9s}   {fname}')
    print('\nChoose: "minimal" (recommended ~46 GB), "all", '
          'or comma-separated indices (e.g. 0,2,5).')
    raw = input('Selection (Enter = minimal): ').strip().lower()
    if raw in ('', 'minimal'):
        return 'minimal'
    if raw == 'all':
        return 'all'
    try:
        idxs = [int(x) for x in raw.split(',') if x.strip() != '']
        return [flist[i][0] for i in idxs]
    except (ValueError, IndexError):
        print('  Could not parse selection, falling back to minimal.')
        return 'minimal'


# ── CLI entry point ─────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description='Download the Griffin dataset from Hugging Face.')
    ap.add_argument('--list', action='store_true', help='print the catalogue and exit')
    ap.add_argument('--subset', default=None, help='subset name (prompts if omitted)')
    ap.add_argument('--minimal', action='store_true', help='download the minimal ~46 GB set')
    ap.add_argument('--all', action='store_true', help='download the entire subset')
    ap.add_argument('--files', default=None,
                    help='comma-separated zip filenames to download')
    ap.add_argument('--dest', default=DEFAULT_DEST, help='download directory')
    ap.add_argument('--no-extract', action='store_true', help='do not unzip after download')
    args = ap.parse_args(argv)

    print('Querying Hugging Face catalogue...')
    catalogue = list_catalogue()

    if args.list:
        print_catalogue(catalogue)
        return

    # Resolve subset: arg -> prompt
    subset = args.subset
    if subset is None:
        subset = _prompt_subset(catalogue)
    elif subset not in catalogue:
        print(f'Unknown subset {subset!r}.')
        print_catalogue(catalogue)
        return

    # Resolve file selection: flags/arg -> prompt
    if args.all:
        files = 'all'
    elif args.minimal:
        files = 'minimal'
    elif args.files:
        files = [f.strip() for f in args.files.split(',') if f.strip()]
    else:
        files = _prompt_files(catalogue, subset)

    result = download_subset(
        subset=subset, files=files, dest=args.dest,
        extract=not args.no_extract, catalogue=catalogue,
    )

    if result['extract_root']:
        try:
            sides = locate_sides(args.dest, subset)
            print('\nExtracted dataset locations:')
            print(f"  dataset root : {sides['dataset_root']}")
            print(f"  vehicle-side : {sides['vehicle_side']}")
            print(f"  drone-side   : {sides['drone_side']}")
        except FileNotFoundError as e:
            print(f'\n[warn] {e}')


if __name__ == '__main__':
    main()
