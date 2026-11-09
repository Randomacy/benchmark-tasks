"""One rigid body made of several box and cylinder shapes, for a yam_arena task file.

yam_arena's task-file objects are single primitives. A ``Compound`` is a task-file Cylinder (its base shape) that also
carries extra shapes, so a bowl can have a handle. A ``Sprung`` part adds a moving piece on a vertical spring-loaded
slide (a two-body articulation), so a clip can have a jaw that must be forced open. Importing this module hooks ``yam_arena.scene`` when it loads, without
changing yam_arena on disk. Everything that needs Isaac Sim is imported only when the object is spawned.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import itertools
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Cylinder, ObjectState


@dataclass(frozen=True)
class Compound(Cylinder):
    """``shapes`` are extra colliders in the base cylinder's frame: ("cyl", (x, y, z), radius, height, colour) or
    ("box", (x, y, z), (sx, sy, sz), colour[, yaw[, tilt]]), with yaw a turn about the vertical and tilt a turn about
    the box's own x axis, in degrees. Those whose first item ends in "_look" have no collider."""

    shapes: tuple = ()
    friction: tuple = (1.2, 1.0)


@dataclass(frozen=True)
class Sprung(Compound):
    """A part with a sprung moving piece: ``shapes`` make the main body and ``moving`` the piece, both given in the
    body frame with the piece at rest. The piece slides straight up from rest by up to ``travel`` (m) and a spring of
    ``rate`` (N/m) pushes it back down. ``pos`` is the body frame's origin, not a centre, and the base cylinder is
    not built. Its state reports the body origin's pose, and ``lift(name)`` how far the piece is pushed up."""

    moving: tuple = ()
    travel: float = 0.01
    rate: float = 500.0
    damping: float = 5.0
    moving_mass: float = 0.005


def _add_shapes(prim_path, shapes, friction, stage, offset=(0.0, 0.0, 0.0), contact=None):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    phys = f"{prim_path}/extra_physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=friction[0], dynamic_friction=friction[1])
    material.func(phys, material)
    for i, shape in enumerate(shapes):
        kind = shape[0]
        centre = Gf.Vec3d(*(float(c) - float(o) for c, o in zip(shape[1], offset)))
        path = f"{prim_path}/extra_{i}"
        if kind.startswith("cyl"):
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(float(shape[2]))
            geom.CreateHeightAttr(float(shape[3]))
            geom.CreateAxisAttr("Z")
            geom.AddTranslateOp().Set(centre)
            colour = shape[4]
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(centre)
            if len(shape) > 4:
                geom.AddRotateZOp().Set(float(shape[4]))
            if len(shape) > 5:
                geom.AddRotateXOp().Set(float(shape[5]))
            geom.AddScaleOp().Set(Gf.Vec3f(*shape[2]))
            colour = shape[3]
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(colour), roughness=0.5)
        look.func(f"{path}_material", look)
        bind_visual_material(path, f"{path}_material", stage=stage)
        if not kind.endswith("_look"):
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            if contact is not None:  # thin contact shells, so mm-scale gaps stay open
                shell = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
                shell.CreateContactOffsetAttr(contact)
                shell.CreateRestOffsetAttr(0.0)
            bind_physics_material(path, phys, stage=stage)


def _corners(shapes):
    """Corners of each shape's box (cylinders as their bounding box), in the frame the shapes are given in."""
    out = []
    for shape in shapes:
        c = np.array(shape[1], dtype=float)
        if shape[0].startswith("cyl"):
            half, yaw, tilt = np.array([shape[2], shape[2], shape[3] / 2]), 0.0, 0.0
        else:
            half = np.array(shape[2], dtype=float) / 2
            yaw = math.radians(shape[4]) if len(shape) > 4 else 0.0
            tilt = math.radians(shape[5]) if len(shape) > 5 else 0.0
        cz, sz, cx, sx = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
        turn = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        for corner in itertools.product((-1, 1), repeat=3):
            out.append(c + turn @ (half * np.array(corner)))
    return np.array(out)


_SPRUNG = {}
_LIFT = {}


def lift(name):
    """How far the named Sprung part's moving piece is pushed up from rest (m), as of the last scoring call."""
    return _LIFT.get(name, 0.0)


def _spawn_sprung(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """The part under ``prim_path``: an Xform holding the main body and the moving piece, joined by a vertical slide
    with a spring drive. The main body is the articulation's root."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    spec = _SPRUNG[cfg.part]
    stage = get_current_stage()
    holder = UsdGeom.Xform.Define(stage, prim_path)
    holder.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    holder.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))

    def rigid(path, mass):
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        tuning = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        tuning.CreateSolverPositionIterationCountAttr(32)
        tuning.CreateMaxDepenetrationVelocityAttr(1.0)
        tuning.CreateAngularDampingAttr(1.0)
        return prim

    body, piece = f"{prim_path}/body", f"{prim_path}/piece"
    root = rigid(body, spec.mass)
    _add_shapes(body, spec.shapes, spec.friction, stage, contact=0.001)
    rigid(piece, spec.moving_mass)
    _add_shapes(piece, spec.moving, spec.friction, stage, contact=0.001)

    slide = UsdPhysics.PrismaticJoint.Define(stage, f"{piece}/slide")
    slide.CreateAxisAttr("Z")
    slide.CreateBody0Rel().SetTargets([Sdf.Path(body)])
    slide.CreateBody1Rel().SetTargets([Sdf.Path(piece)])
    for pos, rot in ((slide.CreateLocalPos0Attr, slide.CreateLocalRot0Attr),
                     (slide.CreateLocalPos1Attr, slide.CreateLocalRot1Attr)):
        pos(Gf.Vec3f(0.0, 0.0, 0.0))
        rot(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    slide.CreateLowerLimitAttr(0.0)
    slide.CreateUpperLimitAttr(float(spec.travel))
    spring = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    spring.CreateTypeAttr("force")
    spring.CreateStiffnessAttr(float(spec.rate))
    spring.CreateDampingAttr(float(spec.damping))
    spring.CreateTargetPositionAttr(0.0)

    UsdPhysics.ArticulationRootAPI.Apply(root)
    solver = PhysxSchema.PhysxArticulationAPI.Apply(root)
    solver.CreateEnabledSelfCollisionsAttr(False)
    solver.CreateSolverPositionIterationCountAttr(32)
    solver.CreateSolverVelocityIterationCountAttr(4)
    return holder.GetPrim()


def _sprung_object(spec):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    _SPRUNG[spec.name] = spec

    @configclass
    class SprungSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_sprung)
        part: str = ""

    class SprungObject(Object):
        def _generate_articulation_cfg(self):
            spring = ImplicitActuatorCfg(joint_names_expr=["slide"], stiffness=spec.rate, damping=spec.damping)
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"spring": spring})
            return self._add_initial_pose_to_cfg(cfg)

    return SprungObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                        spawner_cfg=SprungSpawnerCfg(part=spec.name),
                        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _array(x):
    return getattr(x, "torch", x).detach().cpu().numpy().astype(float)


def _sprung_state(env, spec):
    base = env.unwrapped
    asset = base.scene[spec.name]
    origin = _array(base.scene.env_origins[0])
    pose = _array(asset.data.root_link_pose_w)[0]
    vel = _array(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    x, y, z, w = pose[3:7]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    joints = list(asset.joint_names)
    _LIFT[spec.name] = float(_array(asset.data.joint_pos)[0][joints.index("slide")])
    local = np.vstack([_corners(spec.shapes), _corners(spec.moving) + np.array([0.0, 0.0, _LIFT[spec.name]])])
    world = (R @ local.T).T + (pose[:3] - origin)
    return ObjectState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                       low=world.min(axis=0), high=world.max(axis=0))


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
        plain = [n for n in names if n not in _SPRUNG]
        out = states(env, plain) if plain else {}
        for n in names:
            if n in _SPRUNG:
                out[n] = _sprung_state(env, _SPRUNG[n])
        return {n: out[n] for n in names}

    scene.object_states = object_states

    def task_object(spec):
        if isinstance(spec, Sprung):
            return _sprung_object(spec)
        obj = original(spec)
        if isinstance(spec, Compound):
            from isaaclab.sim.spawners.shapes import spawn_cylinder
            from isaaclab.sim.utils import clone

            @clone
            def spawn(prim_path, cfg, *args, **kwargs):
                prim = spawn_cylinder.__wrapped__(prim_path, cfg, *args, **kwargs)
                _add_shapes(prim_path, spec.shapes, spec.friction, prim.GetStage())
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
