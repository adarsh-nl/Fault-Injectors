"""Snowflake particle sampling - Hahner et al.'s EXACT dart_throwing sampler.

Source: github.com/SysCV/LiDAR_snow_sim  tools/snowfall/sampling.py
(Hahner et al., "LiDAR Snowfall Simulation", CVPR 2022; the same code MultiCorrupt's
snow corruption consumes the output of).

`sekhon_srivastava`, `gunn_marshall` and `dart_throwing` below are copied verbatim
(tqdm progress-bar support is preserved but optional). `ensure_particle_files` is a
thin, seeded driver that reproduces the upstream generation protocol for the three
MultiCorrupt severities: for each severity it samples 64 independent particle sets
(one per LiDAR channel) with Hahner's sampler at R_0 = 80 m using the 'gunn'
distribution, and names the files exactly as MultiCorrupt's runtime expects:

    gunn_{rain_rate}_{occupancy}_{i}.npy      i = 1..64

where rain_rate and occupancy are computed with MultiCorrupt's own
snowfall_rate_to_rainfall_rate / compute_occupancy from the severity's
(snowfall_rate, terminal_velocity) pair (0.5, 1.2) / (2.5, 1.6) / (1.5, 0.4).

Upstream sampled these files once with a shared RNG (seed 42) across a large
parameter product; the published archives are not distributed with MultiCorrupt.
Here each file is generated with an independent, deterministic RNG seeded as
seed + channel_index, which reproduces the same sampling DISTRIBUTION exactly and
makes every file reproducible in isolation.
"""
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:                                     # tqdm optional
    tqdm = None

PI = np.pi

# ============================ BEGIN VERBATIM (Hahner) ============================
def sekhon_srivastava(precipitation_rate: float) -> float:
    """
    :param precipitation_rate:  in mm/h
    :return:                    in 1/cm
    """
    # Determine rate parameter of distribution of snowflake diameters via formula of Sekhon and Srivastava (1970).
    return 22.9 * precipitation_rate ** -0.45


def gunn_marshall(precipitation_rate: float) -> float:
    """
    :param precipitation_rate:  in mm/h
    :return:                    in 1/cm
    """
    # Determine rate parameter of distribution of snowflake diameters via formula of Marshall and Gunn (1958).
    return 25.5 * precipitation_rate ** -0.48


def dart_throwing(occupancy_ratio: float,
                  precipitation_rate: float,
                  R_0: float,
                  rng: np.random.Generator,
                  distribution: str = 'sekhon_srivastava',
                  show_progessbar: bool = False) -> np.ndarray:
    """
    :param occupancy_ratio:     Ratio of the area of the medium occupied by particles.
    :param precipitation_rate:  Measured in millimeters of equivalent liquid water per hour.
    :param R_0:                 Radius of circular disk that forms the domain of sampling (in meters).
    :param rng:                 Random number generator initialized externally with a random seed.
    :param distribution:        Distribition model of particle diameters.
    :param show_progessbar:     Flag if progressbar should be displayed.

    :return:                    N-by-3 array of sampled particles as disks, where each row contains abscissa and
                                ordinate of disk center and disk radius (in meters).
    """

    if distribution == 'sekhon':
        diameter_rate_parameter = sekhon_srivastava(precipitation_rate)
    elif distribution == 'gunn':
        diameter_rate_parameter = gunn_marshall(precipitation_rate)
    else:
        raise NotImplementedError('Distribution model unknown.')

    diameter_scale_parameter = 1 / diameter_rate_parameter              # in cm

    # Initialize samples to empty set.
    samples = np.zeros((0, 3))

    # Initialize occupied area to 0.
    area_occupied = 0.0

    # Calculate global occupied area across entire domain.
    area_occupied_global = occupancy_ratio * PI * R_0 ** 2

    large_number = 1 / occupancy_ratio
    total = area_occupied_global * large_number + 1

    if show_progessbar:

        pbar = tqdm(total=total, desc='sampling particles',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    else:

        pbar = None

    i = 0
    r_avg = 0

    # Main sampling loop.
    while area_occupied < area_occupied_global:

        # Sample center of particle.
        length = np.sqrt(rng.uniform(0, R_0 ** 2))
        angle = rng.uniform(0, 2) * PI

        x = length * np.cos(angle)
        y = length * np.sin(angle)

        particle_diameter = np.inf
        # Sample diameter of particle from exponential distribution (in millimeters).
        while particle_diameter > 20:   # limit diameter to a maximum of 2cm
            particle_diameter = rng.exponential(diameter_scale_parameter * 10)

        # Convert diameter to meters.
        particle_diameter = particle_diameter / 1000

        # Sample height of particle center relative to examined plane.
        height = rng.uniform(-particle_diameter / 2, particle_diameter / 2)

        # Calculate radius of disk that constitutes the intersection of the sampled ball with the examined plane.
        disk_radius = np.sqrt((particle_diameter / 2) ** 2 - height ** 2)

        # If the disk includes the origin, reject the sample and continue.
        if x ** 2 + y ** 2 <= disk_radius ** 2:
            continue

        # Check whether current particle overlaps with any particle that has already been sampled.
        sample_has_overlap = (samples[:, 0] - x) ** 2 + (samples[:, 1] - y) ** 2 <= (samples[:, 2] + disk_radius) ** 2

        # If yes, reject the sample and continue.
        if np.any(sample_has_overlap):
            continue

        else:

            r_avg = (r_avg * i + disk_radius) / (i+1)
            i += 1

            area = PI * disk_radius ** 2
            area_occupied += area
            samples = np.concatenate((samples, np.array([[x, y, disk_radius]])))

            if pbar:
                pbar.update(area * large_number)
                pbar.set_postfix({'n_sampled': len(samples),
                                  'r_avg': r_avg})

    if pbar:
        pbar.n = total
        pbar.close()

    return samples
# ============================= END VERBATIM =============================


# Severity -> (snowfall_rate [mm/h], terminal_velocity [m/s]); MultiCorrupt lidar.py
SEVERITY_SNOWFALL = {1: (0.5, 1.2), 2: (2.5, 1.6), 3: (1.5, 0.4)}
NUM_CHANNELS = 64          # HDL-64 sensor model used by the snow simulation
R_0 = 80.0                 # sampling-domain radius in metres (Hahner sampling.py)


def particle_file_prefix(severity: int) -> str:
    """The exact prefix MultiCorrupt's simulate_snow builds for this severity."""
    from ._mc_snow import snowfall_rate_to_rainfall_rate, compute_occupancy
    s = SEVERITY_SNOWFALL[severity]
    rain_rate = snowfall_rate_to_rainfall_rate(float(s[0]), float(s[1]))
    occupancy = compute_occupancy(float(s[0]), float(s[1]))
    return f'gunn_{rain_rate}_{occupancy}'


def ensure_particle_files(severity: int, npy_dir=None, seed: int = 42,
                          show_progressbar: bool = False, verbose: bool = True):
    """Generate (if missing) the 64 particle files for a severity; returns the prefix.

    Files are written to `npy_dir` (default: <this package>/npy). Existing files are
    kept, so generation is a one-time cost per machine.
    """
    from ._mc_snow import snowfall_rate_to_rainfall_rate, compute_occupancy
    npy_dir = Path(npy_dir) if npy_dir is not None else Path(__file__).parent / 'npy'
    npy_dir.mkdir(parents=True, exist_ok=True)

    s = SEVERITY_SNOWFALL[severity]
    rain_rate = snowfall_rate_to_rainfall_rate(float(s[0]), float(s[1]))
    occupancy = compute_occupancy(float(s[0]), float(s[1]))
    prefix = f'gunn_{rain_rate}_{occupancy}'

    missing = [i for i in range(1, NUM_CHANNELS + 1)
               if not (npy_dir / f'{prefix}_{i}.npy').exists()]
    if missing and verbose:
        print(f'[snowflake_sampling] severity {severity}: generating {len(missing)} '
              f'particle files (one-time; Hahner dart_throwing, R_0={R_0} m) ...')
    for i in missing:
        rng = np.random.default_rng(seed + i)
        particles = dart_throwing(occupancy_ratio=occupancy, precipitation_rate=rain_rate,
                                  R_0=R_0, rng=rng, distribution='gunn',
                                  show_progessbar=show_progressbar)
        np.save(str(npy_dir / f'{prefix}_{i}.npy'), particles)
        if verbose and (i % 16 == 0 or i == missing[-1]):
            print(f'[snowflake_sampling]   {prefix}_{i}.npy  ({len(particles)} flakes)')
    return prefix
