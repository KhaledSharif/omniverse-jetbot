#!/usr/bin/env python3
"""Inspect camera prims in the running Isaac Sim stage.

Run this script INSIDE the Isaac Sim python console or as a standalone
script that queries the running stage.
"""

from pxr import Usd, UsdGeom, Gf
import math

def inspect_cameras_in_stage(stage):
    """Inspect all cameras in a USD stage."""

    print("=" * 60)
    print("CAMERA INSPECTION REPORT")
    print("=" * 60)

    cameras_found = []

    # Traverse all prims looking for cameras
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras_found.append(prim)

    if not cameras_found:
        print("\nNo cameras found in stage!")
        return

    print(f"\nFound {len(cameras_found)} camera(s):\n")

    for cam_prim in cameras_found:
        print("-" * 60)
        camera = UsdGeom.Camera(cam_prim)
        xformable = UsdGeom.Xformable(cam_prim)

        path = str(cam_prim.GetPath())
        name = cam_prim.GetName()

        print(f"Camera: {name}")
        print(f"  Path: {path}")

        # Get world transform
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()

        print(f"  World Position: ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")

        # Get local transform ops
        xform_ops = xformable.GetOrderedXformOps()
        if xform_ops:
            print("  Local Transform Ops:")
            for op in xform_ops:
                op_type = op.GetOpType()
                op_name = op.GetOpName()
                value = op.Get()
                print(f"    - {op_name}: {value}")

        # Get camera properties
        focal_length = camera.GetFocalLengthAttr().Get()
        h_aperture = camera.GetHorizontalApertureAttr().Get()
        v_aperture = camera.GetVerticalApertureAttr().Get()
        clipping = camera.GetClippingRangeAttr().Get()

        print(f"  Focal Length: {focal_length}")
        print(f"  Horizontal Aperture: {h_aperture}")
        print(f"  Vertical Aperture: {v_aperture}")
        if clipping:
            print(f"  Clipping Range: {clipping[0]} - {clipping[1]}")

        # Calculate FOV
        if focal_length and h_aperture:
            fov_h = 2 * math.atan(h_aperture / (2 * focal_length)) * 180 / math.pi
            print(f"  Horizontal FOV: {fov_h:.1f}°")

        print()

    print("=" * 60)
    return cameras_found


if __name__ == "__main__":
    # This is for standalone use - won't work without Isaac Sim running
    print("This script should be run inside Isaac Sim's Python console")
    print("or imported into a running simulation.")
    print()
    print("In Isaac Sim's Script Editor, run:")
    print("  from pxr import Usd")
    print("  import omni.usd")
    print("  stage = omni.usd.get_context().get_stage()")
    print("  exec(open('inspect_stage_cameras.py').read())")
    print("  inspect_cameras_in_stage(stage)")
