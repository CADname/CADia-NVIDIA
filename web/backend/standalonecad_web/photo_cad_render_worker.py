from __future__ import annotations

import argparse
from pathlib import Path


def _set_camera_orientation(cam, orientation: str) -> None:
    o = (orientation or "iso_top_right").lower()
    table = {
        "iso_top_right": ((1, -1, 1), (0, 0, 1)),
        "iso_top_left": ((-1, -1, 1), (0, 0, 1)),
        "iso_bottom_right": ((1, -1, -1), (0, 0, 1)),
        "iso_bottom_left": ((-1, -1, -1), (0, 0, 1)),
        "front": ((0, -1, 0), (0, 0, 1)),
        "back": ((0, 1, 0), (0, 0, 1)),
        "top": ((0, 0, 1), (0, 1, 0)),
        "bottom": ((0, 0, -1), (0, 1, 0)),
        "left": ((-1, 0, 0), (0, 0, 1)),
        "right": ((1, 0, 0), (0, 0, 1)),
    }
    pos, up = table.get(o, table["iso_top_right"])
    cam.SetPosition(*(x * 10 for x in pos))
    cam.SetFocalPoint(0, 0, 0)
    cam.SetViewUp(*up)


def _render_one(polydata, output: Path, orientation: str, width: int, height: int) -> bool:
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtkActor()
    actor.SetMapper(mapper)

    ren = vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.12, 0.13, 0.15)

    win = vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(max(16, min(2048, width)), max(16, min(2048, height)))
    win.AddRenderer(ren)

    cam = ren.GetActiveCamera()
    _set_camera_orientation(cam, orientation)
    ren.ResetCamera()
    try:
        ren.ResetCameraClippingRange()
    except Exception:
        pass
    win.Render()

    filt = vtkWindowToImageFilter()
    filt.SetInput(win)
    filt.SetInputBufferTypeToRGB()
    filt.ReadFrontBufferOff()
    filt.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(str(output))
    writer.SetInputConnection(filt.GetOutputPort())
    writer.Write()

    try:
        win.Finalize()
    except Exception:
        pass
    return output.is_file() and output.stat().st_size > 100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", required=True)
    # Backward-compatible single-view mode.
    parser.add_argument("--output")
    parser.add_argument("--orientation")
    # Preferred multi-view mode: one VTK process, one STL read, four renders.
    parser.add_argument("--output-dir")
    parser.add_argument("--prefix", default="render")
    parser.add_argument("--orientations", default="front,top,right,iso_top_right")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    stl = Path(args.stl).resolve()
    if not stl.is_file():
        raise SystemExit(f"STL not found: {stl}")

    from vtkmodules.vtkIOGeometry import vtkSTLReader

    reader = vtkSTLReader()
    reader.SetFileName(str(stl))
    reader.Update()
    polydata = reader.GetOutput()

    if args.output:
        if not args.orientation:
            raise SystemExit("--orientation is required with --output")
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return 0 if _render_one(polydata, out, args.orientation, args.width, args.height) else 2

    if not args.output_dir:
        raise SystemExit("Provide either --output/--orientation or --output-dir")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    orientations = [x.strip() for x in args.orientations.split(",") if x.strip()]
    if not orientations:
        raise SystemExit("No orientations requested")

    successes = 0
    for orientation in orientations:
        out = out_dir / f"{args.prefix}_{orientation}.png"
        if _render_one(polydata, out, orientation, args.width, args.height):
            successes += 1
    # The parent inspects individual files, matching the previous per-view behaviour.
    # A partial render set is still useful; only zero successful views is an error.
    return 0 if successes > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
