"""
modality.py
-----------
The shared modality gate: which agents of a ``CooperativeSample`` can receive
a fault that needs a given sensor modality.

Written for the transition from homogeneous datasets (OPV2V/V2XSet: every
agent LiDAR-only) to mixed-modality ones (Griffin: LiDAR+camera vehicle
cooperating with a camera-only drone). Two gate strengths exist because those
two worlds need different questions answered:

* :func:`require_all` -- every agent must carry the modality. This is the
  OpenCOOD gate, byte-for-byte: on a homogeneous dataset "some agent lacks it"
  and "the dataset lacks it" coincide, and the strict form also catches a
  malformed sample (an agent that lost its cloud upstream of injection).
* :func:`require_any` + :func:`agents_supporting` -- for mixed datasets: block
  an injector only when NO agent can receive it, and scope it to the agents
  that can. On Griffin, ``require_all(sample, 'lidar')`` would raise on every
  scene (the drone carries no LiDAR *by design*), which conflates "this
  dataset cannot support this injector" with "this agent cannot receive it".

Per-dataset hints: the raised message can carry an adapter-supplied hint
(e.g. OpenCOOD's note that ``retrieve_base_data`` never reads the camera
PNGs), so the shared gate stays dataset-agnostic while the error stays
actionable.
"""

_NEEDS = ('lidar', 'images')


class ModalityError(RuntimeError):
    """Raised when a fault needs a modality the target agents do not carry."""


def _has(agent, need):
    return bool(agent.images) if need == 'images' else agent.lidar is not None


def _check_need(need):
    if need not in _NEEDS:
        raise ValueError("need must be 'lidar' or 'images'")


def agents_supporting(sample, need):
    """Sorted ids of the agents that carry ``need`` -- an injector's scope."""
    _check_need(need)
    return sorted(a for a, ag in sample.agents.items() if _has(ag, need))


def require_all(sample, need, hint=''):
    """Raise ``ModalityError`` unless EVERY agent carries ``need``."""
    _check_need(need)
    missing = sorted(a for a, ag in sample.agents.items()
                     if not _has(ag, need))
    if missing:
        raise ModalityError(
            'fault requires {!r} but agents {} carry none.{}'.format(
                need, missing, (' ' + hint) if hint else ''))


def require_any(sample, need, hint=''):
    """Raise ``ModalityError`` unless AT LEAST ONE agent carries ``need``."""
    _check_need(need)
    if not agents_supporting(sample, need):
        raise ModalityError(
            'fault requires {!r} but no agent in the sample carries it.{}'
            .format(need, (' ' + hint) if hint else ''))
