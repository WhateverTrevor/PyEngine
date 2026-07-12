"""Reusable entity behaviors."""
from __future__ import annotations

import math
import random

import numpy as np
import pygame

from .camera import Camera
from .math3d import Vec3
from .scene import Behavior, Entity


def resolve_collisions(scene, position: Vec3, radius: float) -> Vec3:
    """Push a sphere out of every collidable entity's oriented bounding box.

    Works in each entity's local space (handles rotated walls), so the sphere
    slides along surfaces instead of stopping dead.
    """
    p = position.to_array()
    for _ in range(3):  # a few passes settle corner cases
        moved = False
        for e in scene.entities:
            if e.mesh is None or not e.visible or not e.collidable:
                continue
            model = e.transform.matrix()
            s = e.transform.scale
            reach = e.mesh.bound * max(abs(s.x), abs(s.y), abs(s.z)) + radius
            if np.linalg.norm(p - model[:3, 3]) > reach:
                continue
            try:
                inv = np.linalg.inv(model)
            except np.linalg.LinAlgError:
                continue
            local = inv[:3, :3] @ p + inv[:3, 3]
            closest = np.clip(local, e.mesh.aabb_min, e.mesh.aabb_max)
            world_closest = model[:3, :3] @ closest + model[:3, 3]
            delta = p - world_closest
            dist = float(np.linalg.norm(delta))
            if dist >= radius:
                continue
            if dist < 1e-9:
                # center is inside the box: escape through the nearest face
                pen_lo = local - e.mesh.aabb_min
                pen_hi = e.mesh.aabb_max - local
                axis = int(np.argmin(np.minimum(pen_lo, pen_hi)))
                closest = local.copy()
                closest[axis] = (e.mesh.aabb_min[axis] if pen_lo[axis] < pen_hi[axis]
                                 else e.mesh.aabb_max[axis])
                world_closest = model[:3, :3] @ closest + model[:3, 3]
                out = world_closest - p
                norm = float(np.linalg.norm(out))
                out = out / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])
                p = world_closest + out * radius
            else:
                p = world_closest + delta / dist * radius
            moved = True
        if not moved:
            break
    return Vec3(*p)


class Spin(Behavior):
    """Rotate continuously; speed is radians/second per Euler axis."""

    def __init__(self, speed: Vec3):
        self.speed = speed

    def update(self, entity: Entity, dt: float, engine) -> None:
        r = entity.transform.rotation
        r.x += self.speed.x * dt
        r.y += self.speed.y * dt
        r.z += self.speed.z * dt


class Bob(Behavior):
    """Oscillate vertically around the entity's starting height."""

    def __init__(self, amplitude: float = 0.5, speed: float = 1.0, phase: float = 0.0):
        self.amplitude = amplitude
        self.speed = speed
        self.t = phase

    def start(self, entity: Entity, engine) -> None:
        self.base_y = entity.transform.position.y

    def update(self, entity: Entity, dt: float, engine) -> None:
        self.t += self.speed * dt
        entity.transform.position.y = self.base_y + math.sin(self.t) * self.amplitude


class Orbit(Behavior):
    """Circle around a center point in the XZ plane."""

    def __init__(self, center: Vec3, radius: float, speed: float = 1.0, phase: float = 0.0):
        self.center = center
        self.radius = radius
        self.speed = speed
        self.t = phase

    def update(self, entity: Entity, dt: float, engine) -> None:
        self.t += self.speed * dt
        p = entity.transform.position
        p.x = self.center.x + math.cos(self.t) * self.radius
        p.z = self.center.z + math.sin(self.t) * self.radius


class Flicker(Behavior):
    """Organic torch/faulty-lamp flicker for the entity's light intensity."""

    def __init__(self, amount: float = 0.35, speed: float = 8.0):
        self.amount = amount   # 0..1 fraction of intensity that varies
        self.speed = speed
        self.t = random.uniform(0.0, 100.0)

    def start(self, entity: Entity, engine) -> None:
        self.base = entity.light.intensity if entity.light else 1.0

    def update(self, entity: Entity, dt: float, engine) -> None:
        if entity.light is None:
            return
        self.t += dt
        s, t = self.speed, self.t
        n = (math.sin(t * s) * 0.5
             + math.sin(t * s * 2.7 + 1.7) * 0.35
             + math.sin(t * s * 5.3 + 0.4) * 0.15) * 0.5 + 0.5
        entity.light.intensity = self.base * (1.0 - self.amount + self.amount * n)


class FlyController(Behavior):
    """First-person fly camera with hold-to-look mouse control.

    The mouse is captured only while one of `look_buttons` is held (default:
    left or right button); releasing the button frees the cursor. WASD moves
    along the view direction, Q/E and Space/Ctrl move down/up, Shift is fast.
    `look_guard(mouse_pos)` can veto engaging a look (e.g. over editor UI).
    With `move_requires_look=True` (Unreal viewport behavior), all movement
    keys only act while `looking` is True. While looking, the scroll wheel
    adjusts fly speed instead of whatever it normally controls.
    """

    def __init__(self, camera: Camera, speed: float = 9.0, sensitivity: float = 0.0025,
                 look_buttons: tuple[int, ...] = (1, 3), look_guard=None,
                 collide: bool = True, collide_radius: float = 0.45,
                 move_requires_look: bool = False):
        self.camera = camera
        self.speed = speed
        self.sensitivity = sensitivity
        self.look_buttons = look_buttons
        self.look_guard = look_guard
        self.collide = collide
        self.collide_radius = collide_radius
        self.move_requires_look = move_requires_look
        self.looking = False

    def update(self, entity: Entity, dt: float, engine) -> None:
        inp = engine.input

        if not self.looking:
            for b in self.look_buttons:
                if inp.mouse_button_pressed(b) and (
                        self.look_guard is None or self.look_guard(inp.mouse_pos)):
                    self.looking = True
                    inp.set_captured(True)
                    break
        if self.looking and not any(inp.mouse_held(b) for b in self.look_buttons):
            self.looking = False
            inp.set_captured(False)

        cam = self.camera
        if self.looking:
            cam.yaw -= inp.mouse_dx * self.sensitivity
            cam.pitch -= inp.mouse_dy * self.sensitivity
            limit = math.radians(89.0)
            cam.pitch = max(-limit, min(limit, cam.pitch))
            if inp.wheel:
                self.speed = min(max(self.speed * (1.15 ** inp.wheel), 1.0), 80.0)

        move = Vec3()
        if self.looking or not self.move_requires_look:
            fwd, right = cam.forward(), cam.right()
            if inp.held(pygame.K_w):
                move = move + fwd
            if inp.held(pygame.K_s):
                move = move - fwd
            if inp.held(pygame.K_d):
                move = move + right
            if inp.held(pygame.K_a):
                move = move - right
            if inp.held(pygame.K_SPACE) or inp.held(pygame.K_e):
                move = move + Vec3(0, 1, 0)
            if inp.held(pygame.K_LCTRL) or inp.held(pygame.K_q):
                move = move - Vec3(0, 1, 0)
        if move.length() > 1e-9:
            speed = self.speed * (4.0 if inp.held(pygame.K_LSHIFT) else 1.0)
            cam.position = cam.position + move.normalized() * (speed * dt)
        if self.collide and getattr(engine, "scene", None) is not None:
            cam.position = resolve_collisions(engine.scene, cam.position,
                                              self.collide_radius)


class SunController(Behavior):
    """Drives `scene.light.direction` from this entity's rotation.

    The light travels along the entity's -Z axis -- the same aim convention
    spotlights use -- so the rotate gizmo (E) becomes a time-of-day control:
    spin the Sun entity and the directional light (and its sky disc) follow.
    """

    def update(self, entity: Entity, dt: float, engine) -> None:
        scene = getattr(engine, "scene", None)
        if scene is None:
            return
        m = entity.transform.matrix()
        axis = m[:3, :3] @ np.array([0.0, 0.0, -1.0])
        norm = np.linalg.norm(axis)
        if norm > 1e-9:
            scene.light.direction = Vec3(*(axis / norm))


class FlashlightController(Behavior):
    """Keep the entity's spotlight glued to the camera; toggle with a key."""

    def __init__(self, camera: Camera, toggle_key: int = pygame.K_f):
        self.camera = camera
        self.toggle_key = toggle_key

    def update(self, entity: Entity, dt: float, engine) -> None:
        cam = self.camera
        t = entity.transform
        t.position = Vec3(cam.position.x, cam.position.y, cam.position.z)
        t.rotation.x = cam.pitch
        t.rotation.y = cam.yaw
        if entity.light is not None and engine.input.pressed(self.toggle_key):
            entity.light.enabled = not entity.light.enabled
