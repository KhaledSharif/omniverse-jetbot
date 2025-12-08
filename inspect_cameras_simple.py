#!/usr/bin/env python3
"""
Run this in Isaac Sim's Script Editor (Window > Script Editor)
or paste into the Python console.

Copy this entire script and run it.
"""

import omni.usd
from pxr import Usd, UsdGeom, Gf
import math

stage = omni.usd.get_context().get_stage()

print("\n" + "=" * 70)
print("CAMERA INSPECTION REPORT")
print("=" * 70)

cameras = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        cameras.append(prim)

print(f"\nFound {len(cameras)} camera(s):\n")

for cam_prim in cameras:
    print("-" * 70)
    camera = UsdGeom.Camera(cam_prim)
    xformable = UsdGeom.Xformable(cam_prim)

    path = str(cam_prim.GetPath())
    name = cam_prim.GetName()

    print(f"Camera: {name}")
    print(f"  Path: {path}")

    # World transform
    world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = world_tf.ExtractTranslation()
    print(f"  World Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # Local transform operations
    xform_ops = xformable.GetOrderedXformOps()
    if xform_ops:
        print("  Local Transform Ops:")
        for op in xform_ops:
            val = op.Get()
            print(f"    {op.GetOpName()}: {val}")

    # Camera properties
    fl = camera.GetFocalLengthAttr().Get()
    ha = camera.GetHorizontalApertureAttr().Get()
    va = camera.GetVerticalApertureAttr().Get()
    clip = camera.GetClippingRangeAttr().Get()

    print(f"  Focal Length: {fl}")
    print(f"  Horizontal Aperture: {ha}")
    print(f"  Vertical Aperture: {va}")
    if clip:
        print(f"  Clipping Range: {clip[0]} - {clip[1]}")

    # FOV calculation
    if fl and ha:
        fov = 2 * math.atan(ha / (2 * fl)) * 180 / math.pi
        print(f"  Horizontal FOV: {fov:.1f}°")

    print()

print("=" * 70)
