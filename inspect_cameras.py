#!/usr/bin/env python3
"""Inspect camera prims in the USD stage."""

from pxr import Usd, UsdGeom, Gf
import sys

# Check if USD file path provided
if len(sys.argv) < 2:
    print("Usage: python inspect_cameras.py <path_to_usd_file>")
    sys.exit(1)

usd_path = sys.argv[1]

# Open USD stage
stage = Usd.Stage.Open(usd_path)
if not stage:
    print(f"Failed to open USD file: {usd_path}")
    sys.exit(1)

print(f"Inspecting USD: {usd_path}\n")

# Find all camera prims
cameras = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        cameras.append(prim)

if not cameras:
    print("No cameras found in USD file.")
    sys.exit(0)

print(f"Found {len(cameras)} camera(s):\n")

for cam_prim in cameras:
    camera = UsdGeom.Camera(cam_prim)
    xformable = UsdGeom.Xformable(cam_prim)

    print(f"Camera: {cam_prim.GetPath()}")
    print(f"  Name: {cam_prim.GetName()}")

    # Get transform
    local_transform = xformable.GetLocalTransformation()
    translation = local_transform.ExtractTranslation()
    rotation = local_transform.ExtractRotation()

    print(f"  Position: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})")
    print(f"  Rotation (XYZ Euler): {rotation.GetAxis()} @ {rotation.GetAngle():.1f}°")

    # Get camera properties
    focal_length = camera.GetFocalLengthAttr().Get()
    h_aperture = camera.GetHorizontalApertureAttr().Get()
    v_aperture = camera.GetVerticalApertureAttr().Get()
    clipping = camera.GetClippingRangeAttr().Get()

    print(f"  Focal Length: {focal_length}")
    print(f"  Aperture: {h_aperture} x {v_aperture} mm")
    print(f"  Clipping Range: {clipping}")
    print()
