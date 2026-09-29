"""Resolving the sim/render backend kwargs to concrete, agreeing devices."""

from __future__ import annotations

import torch

from mani_skill.envs.utils.system.backend import (
    parse_backend_device_id,
    sim_backend_name_mapping,
)


def resolve_sim_backend(sim_backend: str | None, num_envs: int) -> str:
    """Turn `None`/"auto"/an alias into the backend name ManiSkill would actually pick.

    Mirrors `BaseEnv.__init__`: "auto" means physx_cuda when running more than one env and
    physx_cpu otherwise. Resolved up front so `pin_backends_to_device` can see what it is deciding
    about, and so the returned env kwargs record the real backend rather than "auto".
    """
    if sim_backend is None:
        sim_backend = "auto"
    if sim_backend == "auto":
        return "physx_cuda" if num_envs > 1 else "physx_cpu"
    name, device_id = parse_backend_device_id(sim_backend)
    name = sim_backend_name_mapping.get(name, name)
    return name if device_id is None else f"{name}:{device_id}"


def backend_kwargs(
    sim_backend: str | None = None,
    render_backend: str | None = None,
    num_envs: int = 1,
) -> dict:
    """`sim_backend`/`render_backend` kwargs with the sapien renderer pinned to the physx device.

    ManiSkill leaves `render_backend` at a bare "gpu", which sapien resolves on its own and which
    on a multi-GPU host need not agree with the sim: `physx_cuda:0` still lets sapien place the
    renderer on cuda:1, and the env then dies with "cuda pose buffer (cuda:0) and the renderer
    (cuda:1) are on different cuda devices". An index-less `physx_cuda` has the same problem one
    level up -- torch resolves "cuda" to cuda:0 while sapien may not -- so it gets an explicit
    index too, from whichever device torch is currently on.

    CPU sim is left alone: it has no cuda pose buffer to disagree with, and forcing sapien_cpu
    would take GPU rendering away from the `physx_cpu` + `obs_mode="rgb"` path, which is the one
    the trajectory replays run on.
    """
    resolved = resolve_sim_backend(sim_backend, num_envs)
    kwargs = {"sim_backend": resolved}
    if render_backend is not None:
        kwargs["render_backend"] = render_backend

    name, device_id = parse_backend_device_id(resolved)
    if name != "physx_cuda" or not torch.cuda.is_available():
        return kwargs
    if device_id is None:
        device_id = torch.cuda.current_device()
        kwargs["sim_backend"] = f"physx_cuda:{device_id}"
    kwargs.setdefault("render_backend", f"sapien_cuda:{device_id}")
    return kwargs
