"""EXACT MultiCorrupt LiDAR corruptions (arXiv:2402.11677), extracted verbatim.

Source: github.com/ika-rwth-aachen/MultiCorrupt  converter/lidar.py
Only the fog / points-reducing / beams-reducing functions are included here;
their bodies are copied unchanged. The two non-algorithmic differences from the
upstream file are:
  * `import pickle` (stdlib) instead of `import pickle5 as pickle` (pickle5 is a
    backport for Python < 3.8; stdlib pickle is protocol-compatible on 3.8+).
  * the snow / misalignment / motion-blur functions and their sklearn/scipy.stats
    imports are omitted (they load nuscenes_snow.yaml at import time; unrelated to fog).
The fog lookup tables must sit in ./fog_lookup_tables next to this file.
"""
import os
import copy
import math
from typing import Dict, List, Tuple
import numpy as np
import pickle
from pathlib import Path
from scipy.constants import speed_of_light as c     # in m/s

PI = np.pi
seed = 1000
np.random.seed(seed)
RNG = np.random.default_rng(seed)

def reduce_LiDAR_beamsV2(pts, severity):
    s = [16, 8, 4][severity - 1]
    
    if s == 16:
        allowed_beams = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    elif s == 8:
        allowed_beams = [1, 5, 9, 13, 17, 21, 25, 29]
    elif s == 4:
        allowed_beams = [1, 9, 17, 25]
    
    mask = np.full(pts.shape[0], False)
    for beam in allowed_beams:
        beam_mask = pts[:, 4] == beam
        mask = np.logical_or(beam_mask, mask)



""" points missing """
def pointsreducing(pts, severity):
    """
    Simulates missing lidar points based on a given severity level.

    Args:
    pts: A numpy array of lidar points.
    severity: An integer between 1 and 3, where 1 is the least severe and 3 is the most severe.

    Returns:
    A numpy array of lidar points with missing points.
    """
    s = [70, 80, 90][severity - 1]

    size = pts.shape[0]
    nr_of_samps = int(round(size * ((100 - s) / 100)))  # Calculate number of points to keep

    permutations = np.random.permutation(size)  # Generate random permutation of indices
    ind = permutations[:nr_of_samps]  # Select indices for points to keep

    pts = pts[ind]  # Extract points based on selected indices

    return pts

'''  fog function'''
INTEGRAL_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / "fog_lookup_tables" 

class ParameterSet:

    def __init__(self, **kwargs) -> None:

        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        ##########################
        # soft target a.k.a. fog #
        ##########################

        # attenuation coefficient => amount of fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # meteorological optical range (in m)
        self.mor = np.log(20) / self.alpha

        # backscattering coefficient (in 1/sr) [sr = steradian]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        ##########
        # sensor #
        ##########

        # pulse peak power (in W)
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # half-power pulse width (in s)
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9

        # total pulse energy (in J)
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # aperture area of the receiver (in in m²)
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # loss of the receiver's optics
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        self.D = 0.1                                    # in m              (displacement of transmitter and receiver)
        self.ROH_T = 0.01                               # in m              (radius of the transmitter aperture)
        self.ROH_R = 0.01                               # in m              (radius of the receiver aperture)
        self.GAMMA_T_DEG = 2                            # in deg            (opening angle of the transmitter's FOV)
        self.GAMMA_R_DEG = 3.5                          # in deg            (opening angle of the receiver's FOV)
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)


        # range at which receiver FOV starts to cover transmitted beam (in m)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # range at which receiver FOV fully covers transmitted beam (in m)
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        ###############
        # hard target #
        ###############

        # distance to hard target (in m)
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # reflectivity of the hard target [0.07, 0.2, > 4 => low, normal, high]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # differential reflectivity of the target
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


def get_integral_dict(p: ParameterSet) -> Dict:
    alpha =p.alpha
    beta=p.beta
    filename = INTEGRAL_PATH / f'integral_0m_to_200m_stepsize_0.1m_alpha_{alpha}_beta_{beta}.pickle'

    with open(filename, 'rb') as handle:
        integral_dict = pickle.load(handle)

    return integral_dict


def P_R_fog_hard(p: ParameterSet, pc: np.ndarray) -> np.ndarray:
    r_0 = np.linalg.norm(pc[:, 0:3], axis=1)
    pc[:, 3] = np.round(np.exp(-2 * p.alpha * r_0) * pc[:, 3])
    return pc


def P_R_fog_soft(p: ParameterSet, pc: np.ndarray, original_intesity: np.ndarray,  noise: int, gain: bool = False,
                 noise_variant: str = 'v1') -> Tuple[np.ndarray, np.ndarray, Dict]:

    augmented_pc = np.zeros(pc.shape)
    fog_mask = np.zeros(len(pc), dtype=bool)

    r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)

    min_fog_response = np.inf
    max_fog_response = 0
    num_fog_responses = 0

    integral_dict = get_integral_dict(p)

    r_noise = RNG.integers(low=1, high=20, size=1)[0]
    r_noise = 10
    for i, r_0 in enumerate(r_zeros):

        # load integral values from precomputed dict
        key = float(str(round(r_0, 1)))
        # limit key to a maximum of 200 m
        fog_distance, fog_response = integral_dict[min(key, 200)]
        fog_response = fog_response * original_intesity[i] * (r_0 ** 2) * p.beta / p.beta_0

        # limit to 255
        # fog_response = min(fog_response, 255)

        if fog_response > pc[i, 3]:

            fog_mask[i] = 1

            num_fog_responses += 1

            scaling_factor = fog_distance / r_0

            augmented_pc[i, 0] = pc[i, 0] * scaling_factor
            augmented_pc[i, 1] = pc[i, 1] * scaling_factor
            augmented_pc[i, 2] = pc[i, 2] * scaling_factor
            augmented_pc[i, 3] = fog_response

            # keep 5th feature if it exists
            if pc.shape[1] > 4:
                augmented_pc[i, 4] = pc[i, 4]

            if noise > 0:

                if noise_variant == 'v1':

                    # add uniform noise based on initial distance
                    distance_noise = RNG.uniform(low=r_0 - noise, high=r_0 + noise, size=1)[0]
                    noise_factor = r_0 / distance_noise

                elif noise_variant == 'v2':

                    # add noise in the power domain
                    power = RNG.uniform(low=-1, high=1, size=1)[0]
                    noise_factor = max(1.0, noise/5) ** power       # noise=10 => noise_factor ranges from 1/2 to 2

                elif noise_variant == 'v3':

                    # add noise in the power domain
                    power = RNG.uniform(low=-0.5, high=1, size=1)[0]
                    noise_factor = max(1.0, noise*4/10) ** power    # noise=10 => ranges from 1/2 to 4

                elif noise_variant == 'v4':

                    additive = r_noise * RNG.beta(a=2, b=20, size=1)[0]
                    new_dist = fog_distance + additive
                    noise_factor = new_dist / fog_distance

                else:

                    raise NotImplementedError(f"noise variant '{noise_variant}' is not implemented (yet)")

                augmented_pc[i, 0] = augmented_pc[i, 0] * noise_factor
                augmented_pc[i, 1] = augmented_pc[i, 1] * noise_factor
                augmented_pc[i, 2] = augmented_pc[i, 2] * noise_factor

            if fog_response > max_fog_response:
                max_fog_response = fog_response

            if fog_response < min_fog_response:
                min_fog_response = fog_response

        else:
            augmented_pc[i] = pc[i]

    if gain:
        max_intensity = np.ceil(max(augmented_pc[:, 3]))
        gain_factor = 255 / max_intensity
        augmented_pc[:, 3] *= gain_factor

    simulated_fog_pc = None
    num_fog = 0
    if num_fog_responses > 0:
        fog_points = augmented_pc[fog_mask]
        simulated_fog_pc = fog_points
        num_fog = len(fog_points)


    info_dict = {'min_fog_response': min_fog_response,
                 'max_fog_response': max_fog_response,
                 'num_fog_responses': num_fog_responses,}

    return augmented_pc, simulated_fog_pc,  num_fog, info_dict


def simulate_fog(severity, pc: np.ndarray, noise: int, gain: bool = False, noise_variant: str = 'v1',
                 hard: bool = True, soft: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    s = [(0.02,0.008) ,(0.03,0.008),(0.06,0.05)][severity - 1]
    p = ParameterSet(alpha=s[0],beta=s[1]) 
    augmented_pc = copy.deepcopy(pc)
    original_intensity = copy.deepcopy(pc[:, 3])

    info_dict = None
    simulated_fog_pc = None

    if hard:
        augmented_pc = P_R_fog_hard(p, augmented_pc)
    if soft:
        augmented_pc, simulated_fog_pc,  num_fog, info_dict = P_R_fog_soft(p, augmented_pc, original_intensity,  noise, gain,
                                                                 noise_variant)

    return augmented_pc, simulated_fog_pc, num_fog, info_dict
