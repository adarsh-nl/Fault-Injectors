"""
pcd.py
------
Minimal dependency-free PCD (Point Cloud Data) reader.

Supports ASCII and binary encodings, which covers OPV2V / V2XSet / DAIR-V2X
point clouds, without pulling in open3d. `binary_compressed` files raise a
clear error (convert once with open3d/pypcd if you have them).
"""

import numpy as np

_PCD_TO_NUMPY = {
    ('F', 4): 'f4', ('F', 8): 'f8',
    ('I', 1): 'i1', ('I', 2): 'i2', ('I', 4): 'i4', ('I', 8): 'i8',
    ('U', 1): 'u1', ('U', 2): 'u2', ('U', 4): 'u4', ('U', 8): 'u8',
}


def load_pcd(path, columns=('x', 'y', 'z', 'intensity')):
    """
    Read a .pcd file and return an (N, len(columns)) float32 array.

    Requested columns absent from the file are filled with zeros (some
    exports omit intensity). Extra columns in the file are ignored.

    Intensity aliasing: OPV2V/V2XSet .pcd files carry no field literally
    named ``intensity`` -- their header is ``FIELDS x y z rgb``, with the
    return intensity stored in the RED byte of a bit-packed PCL rgb float
    (open3d exposes it as ``colors[:, 0]``, which is exactly what OpenCOOD's
    ``pcd_utils.pcd_to_np`` reads). Before 2026-08-05 this function matched
    field names literally, found no ``intensity``, and silently zero-filled
    the column -- which is why LidarFog/LidarSnow appeared to no-op on these
    clouds. A requested ``intensity`` column now falls back to an ``I``/``i``
    field (Griffin-style) and then to unpacking ``rgb``; the unpack is
    bit-identical to ``pcd_to_np`` (verified max|d| = 0.0 on V2XSet).
    """
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition(' ')
            header[key.upper()] = value
            if key.upper() == 'DATA':
                break

        fields = header['FIELDS'].split()
        sizes  = [int(s) for s in header['SIZE'].split()]
        types  = header['TYPE'].split()
        counts = [int(c) for c in header.get('COUNT', ' '.join(['1'] * len(fields))).split()]
        n_pts  = int(header.get('POINTS', header.get('WIDTH', '0')))
        data   = header['DATA'].lower()

        names, formats = [], []
        for fname, size, typ, count in zip(fields, sizes, types, counts):
            base = _PCD_TO_NUMPY.get((typ, size))
            if base is None:
                raise ValueError(f'{path}: unsupported PCD field type {typ}{size}')
            if count == 1:
                names.append(fname)
                formats.append(base)
            else:
                for i in range(count):
                    names.append(f'{fname}_{i}')
                    formats.append(base)
        dtype = np.dtype({'names': names, 'formats': formats})

        if data == 'ascii':
            raw = np.loadtxt(f, dtype=np.float64, max_rows=n_pts, ndmin=2)
            rec = np.zeros(len(raw), dtype=dtype)
            for i, name in enumerate(names):
                rec[name] = raw[:, i]
        elif data == 'binary':
            rec = np.frombuffer(f.read(n_pts * dtype.itemsize), dtype=dtype,
                                count=n_pts)
        else:
            raise ValueError(
                f'{path}: PCD encoding {data!r} is not supported '
                f'(ascii and binary are). Re-export with open3d if needed.')

    out = np.zeros((len(rec), len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        if col in names:
            out[:, j] = rec[col].astype(np.float32)
        elif col == 'intensity':
            alias = next((a for a in ('I', 'i') if a in names), None)
            if alias is not None:
                out[:, j] = rec[alias].astype(np.float32)
            elif 'rgb' in names:
                # PCL packed-rgb float: red byte = bits 16-23. The cast to
                # float32 first is exact (the packed value was written as
                # float32), so the bit view recovers the original bytes.
                bits = np.ascontiguousarray(
                    rec['rgb'].astype(np.float32)).view(np.uint32)
                out[:, j] = ((bits >> 16) & 0xFF).astype(np.float32) / 255.0
    return out
