"""One rigid body made of several box and cylinder shapes, for a yam_arena task file.

yam_arena's task-file objects are single primitives. A ``Compound`` is a task-file Cylinder (its base shape) that also
carries extra shapes, so a bowl can have a handle. Importing this module hooks ``yam_arena.scene`` when it loads, without
changing yam_arena on disk. Everything that needs Isaac Sim is imported only when the object is spawned.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass

from yam_arena.task_file import Cylinder


@dataclass(frozen=True)
class Compound(Cylinder):
    """``shapes`` are extra colliders in the base cylinder's frame: ("cyl", (x, y, z), radius, height, colour) or
    ("box", (x, y, z), (sx, sy, sz), colour). Those whose first item ends in "_look" have no collider."""

    shapes: tuple = ()
    friction: tuple = (1.2, 1.0)


def _add_shapes(prim_path, spec, stage):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    phys = f"{prim_path}/extra_physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    material.func(phys, material)
    for i, shape in enumerate(spec.shapes):
        kind, centre = shape[0], shape[1]
        path = f"{prim_path}/extra_{i}"
        if kind.startswith("cyl"):
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(float(shape[2]))
            geom.CreateHeightAttr(float(shape[3]))
            geom.CreateAxisAttr("Z")
            geom.AddTranslateOp().Set(Gf.Vec3d(*centre))
            colour = shape[4]
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*centre))
            geom.AddScaleOp().Set(Gf.Vec3f(*shape[2]))
            colour = shape[3]
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(colour), roughness=0.5)
        look.func(f"{path}_material", look)
        bind_visual_material(path, f"{path}_material", stage=stage)
        if not kind.endswith("_look"):
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            bind_physics_material(path, phys, stage=stage)


_ENV = {"env": None}


def gripper_points():
    """Where each arm's finger links are (station frame, env 0), or None before the first scoring step."""
    env = _ENV["env"]
    if env is None:
        return None
    base = env.unwrapped
    origin = base.scene.env_origins[0].detach().cpu().numpy()
    points = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        pos = robot.data.body_pos_w[0].detach().cpu().numpy() - origin
        points += [pos[i] for i, name in enumerate(robot.body_names) if any(k in name.lower() for k in FINGER_WORDS)]
    return points


FINGER_WORDS = ("finger", "tip", "link_7", "link_8", "tcp", "gripper")


def _patch(scene):
    original = scene.task_object
    states = scene.object_states

    def object_states(env, names):
        _ENV["env"] = env
        return states(env, names)

    scene.object_states = object_states

    def task_object(spec):
        obj = original(spec)
        if isinstance(spec, Compound):
            from isaaclab.sim.spawners.shapes import spawn_cylinder
            from isaaclab.sim.utils import clone

            @clone
            def spawn(prim_path, cfg, *args, **kwargs):
                prim = spawn_cylinder.__wrapped__(prim_path, cfg, *args, **kwargs)
                _add_shapes(prim_path, spec, prim.GetStage())
                return prim

            obj.spawner_cfg.func = spawn
            obj.object_cfg.spawn.func = spawn  # the object copied its spawn settings when it was created
        return obj

    scene.task_object = task_object


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if "yam_arena.scene" in sys.modules:
    _patch(sys.modules["yam_arena.scene"])
elif not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
