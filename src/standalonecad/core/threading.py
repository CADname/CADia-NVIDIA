from __future__ import annotations

"""General modeled-thread definition and application layer.

The architecture mirrors the public Autodesk Inventor API separation between
thread information (ThreadInfo) and a feature/application that consumes it,
without depending on Inventor itself:

* :class:`ThreadDefinition` owns the semantic thread profile.
* :class:`ThreadApplication` owns where/how that definition is applied.
* external and internal modeled geometry are generated from the same definition.

This module is intentionally generic.  It is not tied to one prompt, one lead
screw, or one requested angle.
"""

from dataclasses import dataclass
import math
from typing import Any

import cadquery as cq


def _finite_positive(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _finite_nonnegative(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


@dataclass(frozen=True)
class ThreadDefinition:
    """Role-neutral thread profile, analogous to Inventor ``ThreadInfo``.

    ``form`` describes the semantic profile.  The same definition can be used
    by an external and an internal application, which prevents mating geometry
    from silently diverging.
    """

    form: str
    nominal_diameter_mm: float
    pitch_mm: float
    starts: int = 1
    included_angle_deg: float = 60.0
    right_handed: bool = True
    designation: str | None = None
    thread_type: str | None = None
    thread_class: str | None = None
    crest_flat_fraction: float | None = None
    radial_depth_mm: float | None = None

    def __post_init__(self) -> None:
        form = str(self.form or "").strip().lower().replace("-", "_")
        if form not in {"iso_metric", "trapezoidal", "custom"}:
            raise ValueError("thread form must be iso_metric|trapezoidal|custom")
        diameter = _finite_positive("nominal_diameter_mm", self.nominal_diameter_mm)
        pitch = _finite_positive("pitch_mm", self.pitch_mm)
        starts = int(self.starts)
        if starts < 1 or starts > 8:
            raise ValueError("starts must be between 1 and 8")
        angle = float(self.included_angle_deg)
        if not math.isfinite(angle) or not (5.0 <= angle < 175.0):
            raise ValueError("included_angle_deg must be >= 5 and < 175")
        if form == "iso_metric" and abs(angle - 60.0) > 1e-9:
            raise ValueError("iso_metric thread form requires a 60 degree included angle")
        if self.crest_flat_fraction is not None:
            fraction = float(self.crest_flat_fraction)
            if not math.isfinite(fraction) or not (0.0 < fraction < 0.9):
                raise ValueError("crest_flat_fraction must be between 0 and 0.9")
        if self.radial_depth_mm is not None:
            _finite_positive("radial_depth_mm", self.radial_depth_mm)
        object.__setattr__(self, "form", form)
        object.__setattr__(self, "nominal_diameter_mm", diameter)
        object.__setattr__(self, "pitch_mm", pitch)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "included_angle_deg", angle)
        object.__setattr__(self, "right_handed", bool(self.right_handed))

    @classmethod
    def iso_metric(
        cls,
        diameter_mm: float,
        pitch_mm: float,
        right_handed: bool = True,
        *,
        designation: str | None = None,
        thread_class: str | None = None,
        crest_flat_fraction: float = 0.125,
    ) -> "ThreadDefinition":
        return cls(
            form="iso_metric",
            nominal_diameter_mm=diameter_mm,
            pitch_mm=pitch_mm,
            starts=1,
            included_angle_deg=60.0,
            right_handed=right_handed,
            designation=designation or f"M{float(diameter_mm):g}x{float(pitch_mm):g}",
            thread_type="ISO Metric Profile",
            thread_class=thread_class,
            crest_flat_fraction=crest_flat_fraction,
        )

    @classmethod
    def trapezoidal(
        cls,
        major_diameter_mm: float,
        pitch_mm: float,
        starts: int = 1,
        included_angle_deg: float = 30.0,
        right_handed: bool = True,
        *,
        designation: str | None = None,
    ) -> "ThreadDefinition":
        # Keep the pre-existing CADia lead-screw contract while moving the
        # geometry into the shared definition/application architecture.
        angle = float(included_angle_deg)
        if not (20.0 <= angle <= 60.0):
            raise ValueError("trapezoidal included_angle_deg must be between 20 and 60")
        return cls(
            form="trapezoidal",
            nominal_diameter_mm=major_diameter_mm,
            pitch_mm=pitch_mm,
            starts=starts,
            included_angle_deg=angle,
            right_handed=right_handed,
            designation=designation,
        )

    @property
    def lead_mm(self) -> float:
        return self.pitch_mm * self.starts

    def radial_depth(self, role: str = "external") -> float:
        role = str(role).strip().lower()
        if role not in {"external", "internal"}:
            raise ValueError("thread role must be external|internal")
        if self.radial_depth_mm is not None:
            return min(float(self.radial_depth_mm), self.nominal_diameter_mm * 0.22)
        if self.form == "iso_metric":
            # Preserve the existing modeled ISO approximations while sharing
            # the same semantic definition.  Internal/external truncation may
            # differ even when the standard thread designation is the same.
            factor = 0.61343 if role == "external" else 0.541265
            return min(factor * self.pitch_mm, self.nominal_diameter_mm * 0.22)
        return min(0.45 * self.pitch_mm, self.nominal_diameter_mm * 0.22)

    def profile_dimensions(self, role: str = "external") -> dict[str, float]:
        depth = self.radial_depth(role)
        half_root = min(0.45 * self.pitch_mm, self.pitch_mm * 0.36)
        if self.crest_flat_fraction is not None:
            crest = max(0.02 * self.pitch_mm, min(0.35 * self.pitch_mm, float(self.crest_flat_fraction) * self.pitch_mm))
            half_crest = crest / 2.0
        else:
            # Included flank angle drives the same axial profile for both
            # mating roles.  This is the key invariant that the previous
            # separate screw/nut generators did not enforce.
            half_crest = max(
                0.06 * self.pitch_mm,
                half_root - depth * math.tan(math.radians(self.included_angle_deg / 2.0)),
            )
            half_crest = max(0.02 * self.pitch_mm, min(half_root * 0.9, half_crest))
        return {
            "depth_mm": depth,
            "half_root_mm": half_root,
            "half_crest_mm": half_crest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "nominal_diameter_mm": self.nominal_diameter_mm,
            "major_diameter_mm": self.nominal_diameter_mm,
            "pitch_mm": self.pitch_mm,
            "lead_mm": self.lead_mm,
            "starts": self.starts,
            "included_angle_deg": self.included_angle_deg,
            # Backward-compatible name used by the existing accelerator.
            "thread_angle_deg": self.included_angle_deg,
            "right_handed": self.right_handed,
            "designation": self.designation,
            "thread_type": self.thread_type,
            "thread_class": self.thread_class,
        }


@dataclass(frozen=True)
class ThreadApplication:
    """Application of a :class:`ThreadDefinition` to modeled geometry.

    This mirrors the Inventor distinction between reusable thread information
    and the ThreadFeature that applies it to a particular face/extent.
    """

    definition: ThreadDefinition
    internal: bool
    length_mm: float
    z0_mm: float = 0.0
    clearance_mm: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_mm", _finite_positive("length_mm", self.length_mm))
        z0 = float(self.z0_mm)
        if not math.isfinite(z0):
            raise ValueError("z0_mm must be finite")
        object.__setattr__(self, "z0_mm", z0)
        object.__setattr__(self, "clearance_mm", _finite_nonnegative("clearance_mm", self.clearance_mm))
        object.__setattr__(self, "internal", bool(self.internal))

    @property
    def role(self) -> str:
        return "internal" if self.internal else "external"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "length_mm": self.length_mm,
            "z0_mm": self.z0_mm,
            "clearance_mm": self.clearance_mm,
            "definition": self.definition.to_dict(),
        }


def _swept_ridges(
    application: ThreadApplication,
    *,
    base_radius: float,
    depth: float,
    half_root: float,
    half_crest: float,
    radial_overlap: float,
) -> list[cq.Shape]:
    definition = application.definition
    z0 = application.z0_mm
    profile = (
        cq.Workplane("XZ", origin=(0, 0, z0))
        .center(base_radius, 0)
        .polyline([
            (-radial_overlap, -half_root),
            (depth, -half_crest),
            (depth, half_crest),
            (-radial_overlap, half_root),
        ])
        .close().wire().val()
    )
    ridges: list[cq.Shape] = []
    for index in range(definition.starts):
        helix = cq.Wire.makeHelix(
            definition.lead_mm,
            application.length_mm,
            base_radius,
            center=(0, 0, z0),
            dir=(0, 0, 1),
            lefthand=not definition.right_handed,
        )
        ridge = cq.Solid.sweep(profile, [], helix, makeSolid=True, isFrenet=True)
        if index:
            ridge = ridge.rotate((0, 0, 0), (0, 0, 1), 360.0 * index / definition.starts)
        ridges.append(ridge)
    return ridges


def _metadata(application: ThreadApplication, *, minor_radius: float, dims: dict[str, float]) -> dict[str, Any]:
    definition = application.definition
    return {
        "major_diameter_mm": definition.nominal_diameter_mm,
        "minor_diameter_mm": 2.0 * minor_radius,
        "pitch_mm": definition.pitch_mm,
        "lead_mm": definition.lead_mm,
        "starts": definition.starts,
        "thread_angle_deg": definition.included_angle_deg,
        "length_mm": application.length_mm,
        "thread_length_mm": application.length_mm,
        "thread_depth_mm": dims["depth_mm"],
        "profile_half_root_mm": dims["half_root_mm"],
        "profile_half_crest_mm": dims["half_crest_mm"],
        "clearance_mm": application.clearance_mm,
        "thread_role": application.role,
        "thread_definition": definition.to_dict(),
        "thread_application": application.to_dict(),
    }


def build_modeled_thread(application: ThreadApplication) -> tuple[cq.Shape, dict[str, Any]]:
    """Build external thread solid or internal thread cutting solid.

    The internal and external paths consume the exact same ThreadDefinition.
    Only role-specific standard truncation and clearance differ.
    """

    definition = application.definition
    dims = definition.profile_dimensions(application.role)
    depth = dims["depth_mm"]
    r_major = definition.nominal_diameter_mm / 2.0
    r_minor = max(0.05, r_major - depth)
    overlap = min(0.08, max(0.01, depth * 0.08))

    if not application.internal:
        axial_overlap = min(0.10, max(0.02, definition.pitch_mm * 0.05)) if application.z0_mm > 0 else 0.0
        base = cq.Solid.makeCylinder(
            r_minor + overlap / 2.0,
            application.length_mm + axial_overlap,
            (0, 0, application.z0_mm - axial_overlap),
            (0, 0, 1),
        )
        ridges = _swept_ridges(
            application,
            base_radius=r_minor,
            depth=depth,
            half_root=dims["half_root_mm"],
            half_crest=dims["half_crest_mm"],
            radial_overlap=overlap,
        )
        result: cq.Shape = base
        for ridge in ridges:
            try:
                result = cq.Workplane(obj=result).union(ridge).val()
            except Exception:
                result = cq.Compound.makeCompound([result, ridge])
        clip = cq.Solid.makeCylinder(
            r_major + 0.10,
            application.length_mm,
            (0, 0, application.z0_mm),
            (0, 0, 1),
        )
        try:
            result = cq.Workplane(obj=result).intersect(clip).val()
        except Exception:
            pass
        return result, _metadata(application, minor_radius=r_minor, dims=dims)

    # Internal application returns a cutting solid.  Clearance translates the
    # profile radially without changing its included flank angle, so mating
    # internal/external profiles continue to share one semantic definition.
    base_radius = r_minor + application.clearance_mm
    core = cq.Solid.makeCylinder(
        base_radius,
        application.length_mm + 0.2,
        (0, 0, application.z0_mm - 0.1),
        (0, 0, 1),
    )
    grooves = _swept_ridges(
        application,
        base_radius=base_radius,
        depth=depth,
        half_root=dims["half_root_mm"],
        half_crest=dims["half_crest_mm"],
        radial_overlap=overlap,
    )
    result = core
    for groove in grooves:
        try:
            result = cq.Workplane(obj=result).union(groove).val()
        except Exception:
            result = cq.Compound.makeCompound([result, groove])
    return result, _metadata(application, minor_radius=r_minor, dims=dims)
