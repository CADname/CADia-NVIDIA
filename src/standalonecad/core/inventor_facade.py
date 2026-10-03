from __future__ import annotations

"""Inventor-style object façade over the existing StandaloneCAD backend.

This module is intentionally additive.  It does not replace the existing document,
feature, assembly, solver, recovery, or strict ipt-mcp command paths.  Instead it
provides an object model shaped like Inventor's ComponentDefinition / Occurrences /
ComponentOccurrence / geometry-proxy concepts while delegating all real modeling to
the already-tested StandaloneCAD backend.
"""

from dataclasses import dataclass
import base64
import copy
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .assembly import Occurrence, find_interface, transformed_frame, transform_shape
from .joints import _intent_local_frame, _transform_local_frame
from .topology import make_face_ref


def _encode_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_key(value: str) -> dict[str, Any]:
    s = str(value)
    s += "=" * ((4 - len(s) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(s.encode("ascii"))
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid standalone reference key") from exc
    if not isinstance(obj, dict):
        raise ValueError("Invalid standalone reference key payload")
    return obj


@dataclass(frozen=True)
class GeometryProxy:
    """Assembly-context proxy for a component-local reference.

    ``entity`` is either a named work/origin/iMate reference string or a
    GeometryIntent-like dict already understood by the joint backend.
    """

    occurrence: "ComponentOccurrence"
    entity: str | dict[str, Any]

    @property
    def occurrence_name(self) -> str:
        return self.occurrence.name

    @property
    def native_object(self) -> str | dict[str, Any]:
        return copy.deepcopy(self.entity)

    @property
    def is_named_reference(self) -> bool:
        return isinstance(self.entity, str)

    @property
    def reference_name(self) -> str | None:
        return self.entity if isinstance(self.entity, str) else None

    @property
    def geometry_intent(self) -> dict[str, Any] | None:
        return None if isinstance(self.entity, str) else copy.deepcopy(self.entity)

    def world_frame(self) -> dict[str, Any]:
        occ = self.occurrence._native
        if isinstance(self.entity, str):
            item = find_interface(occ.interfaces, self.entity)
            p, d = transformed_frame(item.get("origin", [0, 0, 0]), item.get("direction"), occ)
            return {
                "origin": [float(x) for x in p],
                "direction": None if d is None else [float(x) for x in d],
                "x_direction": None,
            }
        p, z, x, _ = _intent_local_frame(occ, occ.interfaces, None, self.entity)
        p, z, x = _transform_local_frame(p, z, x, occ)
        return {
            "origin": [float(v) for v in p],
            "direction": None if z is None else [float(v) for v in z],
            "x_direction": None if x is None else [float(v) for v in x],
        }

    def reference_key(self) -> str:
        return self.occurrence.parent.reference_manager.get_reference_key(self)


class ComponentOccurrence:
    """Inventor-like wrapper around one native assembly Occurrence."""

    def __init__(self, parent: "AssemblyComponentDefinition", native: Occurrence, creation_result: dict[str, Any] | None = None):
        self.parent = parent
        self._native = native
        self._creation_result = copy.deepcopy(creation_result) if isinstance(creation_result, dict) else None

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def path(self) -> str:
        return self._native.path

    @property
    def grounded(self) -> bool:
        return bool(self._native.grounded)

    @grounded.setter
    def grounded(self, value: bool) -> None:
        self._native.grounded = bool(value)
        self.parent.document.rebuild_assembly()

    @property
    def suppressed(self) -> bool:
        return bool(self._native.suppressed)

    @suppressed.setter
    def suppressed(self, value: bool) -> None:
        self._native.suppressed = bool(value)
        self.parent.document.rebuild_assembly()

    @property
    def position_mm(self) -> list[float]:
        return list(self._native.position_mm)

    @property
    def rotation_deg_xyz(self) -> list[float]:
        return list(self._native.rotation_deg_xyz)

    @property
    def transformation(self) -> dict[str, list[float]]:
        return {"position_mm": self.position_mm, "rotation_deg_xyz": self.rotation_deg_xyz}

    @property
    def component_type(self) -> str:
        return str(getattr(self._native, "component_type", "part"))

    @property
    def native_object(self) -> Occurrence:
        return self._native

    def set_transform(self, position_mm: list[float] | None = None, rotation_deg_xyz: list[float] | None = None) -> "ComponentOccurrence":
        if position_mm is not None:
            if len(position_mm) != 3:
                raise ValueError("position_mm must contain 3 values")
            self._native.position_mm = [float(x) for x in position_mm]
        if rotation_deg_xyz is not None:
            if len(rotation_deg_xyz) != 3:
                raise ValueError("rotation_deg_xyz must contain 3 values")
            self._native.rotation_deg_xyz = [float(x) for x in rotation_deg_xyz]
        self.parent.document.rebuild_assembly()
        return self

    def create_geometry_proxy(self, geometry: str | dict[str, Any]) -> GeometryProxy:
        return self.parent.reference_manager.create_geometry_proxy(self, geometry)

    def world_shape(self):
        return transform_shape(self._native.shape, self._native.position_mm, self._native.rotation_deg_xyz)

    def reference_key(self) -> str:
        return self.parent.reference_manager.get_reference_key(self)

    def creation_result(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._creation_result)

    def delete(self):
        """Delete this occurrence from the assembly, preserving its source document."""
        return self.parent.occurrences.delete(self.name)

    def replace(self, full_document_name: str, replace_all: bool = False):
        """Inventor ComponentOccurrence.Replace-style component replacement."""
        return self.parent.document.replace_occurrence(self.name, full_document_name, replace_all)

    def suppress(self):
        self.suppressed = True
        return self

    def unsuppress(self):
        self.suppressed = False
        return self


class ComponentOccurrences:
    """Collection wrapper matching Inventor's occurrence-centric access pattern."""

    def __init__(self, parent: "AssemblyComponentDefinition"):
        self.parent = parent

    def __len__(self) -> int:
        return len(self.parent.document.occurrences)

    def __iter__(self) -> Iterator[ComponentOccurrence]:
        for occ in self.parent.document.occurrences.values():
            yield ComponentOccurrence(self.parent, occ)

    def item(self, key: int | str) -> ComponentOccurrence:
        if isinstance(key, int):
            if key < 1:
                raise IndexError("Occurrence index is 1-based")
            try:
                native = list(self.parent.document.occurrences.values())[key - 1]
            except IndexError as exc:
                raise IndexError(f"Occurrence index out of range: {key}") from exc
            return ComponentOccurrence(self.parent, native)
        name = str(key)
        native = self.parent.document.occurrences.get(name)
        if native is None:
            raise KeyError(f"Occurrence not found: {name}")
        return ComponentOccurrence(self.parent, native)

    def __getitem__(self, key: int | str) -> ComponentOccurrence:
        return self.item(key)

    def add(self, full_document_name: str, position: dict[str, Any] | None = None, grounded: bool = False) -> ComponentOccurrence:
        pos = None; rot = None
        if position is not None:
            if not isinstance(position, dict):
                raise ValueError("position must be a dict containing position_mm and/or rotation_deg_xyz")
            pos = position.get("position_mm")
            rot = position.get("rotation_deg_xyz")
        result = self.parent.document.place_occurrence(full_document_name, grounded, pos, rot)
        native = self.parent.document.occurrences[result["occurrence_name"]]
        return ComponentOccurrence(self.parent, native, result)

    def names(self) -> list[str]:
        return list(self.parent.document.occurrences)

    def delete(self, name: str):
        return self.parent.document.delete_occurrence(str(name))


class AssemblyConstraints:
    """Object collection façade delegating to the existing v10.6 constraint backend."""

    def __init__(self, parent: "AssemblyComponentDefinition"):
        self.parent = parent

    def __iter__(self):
        return iter(self.parent.document.constraints)

    def __len__(self) -> int:
        return len(self.parent.document.constraints)

    def item(self, key: int | str):
        if isinstance(key,int):
            if key < 1:raise IndexError('Constraint index is 1-based')
            try:return self.parent.document.constraints[key-1]
            except IndexError as exc:raise IndexError(f'Constraint index out of range: {key}') from exc
        name=str(key); found=next((x for x in self.parent.document.constraints if x.name==name),None)
        if found is None:raise KeyError(f'Constraint not found: {name}')
        return found

    def delete(self, name: str):
        return self.parent.document.delete_assembly_constraint(str(name))

    @staticmethod
    def _side(value: ComponentOccurrence | GeometryProxy | None, ref: str | None):
        if isinstance(value, GeometryProxy):
            if not value.is_named_reference:
                raise ValueError("Legacy AssemblyConstraint currently requires a named-reference GeometryProxy")
            return value.occurrence_name, value.reference_name
        if isinstance(value, ComponentOccurrence):
            if not ref:
                raise ValueError("A named reference is required with ComponentOccurrence")
            return value.name, ref
        return None, ref

    def add(self, type_: str, a: ComponentOccurrence | GeometryProxy | None, a_ref: str | None,
            b: ComponentOccurrence | GeometryProxy | None, b_ref: str | None,
            offset_mm: float = 0.0, angle_deg: float | None = None, insert_opposed: bool = True):
        a_occ, a_name = self._side(a, a_ref); b_occ, b_name = self._side(b, b_ref)
        if not a_name or not b_name:
            raise ValueError("AssemblyConstraint requires references on both sides")
        return self.parent.document.add_assembly_constraint(
            type_, a_occ, a_name, b_occ, b_name, offset_mm, angle_deg, insert_opposed
        )

    def add_mate(self, a, a_ref, b, b_ref, offset_mm: float = 0.0):
        return self.add("mate", a, a_ref, b, b_ref, offset_mm=offset_mm)

    def add_flush(self, a, a_ref, b, b_ref, offset_mm: float = 0.0):
        return self.add("flush", a, a_ref, b, b_ref, offset_mm=offset_mm)

    def add_insert(self, a, a_ref, b, b_ref, offset_mm: float = 0.0, opposed: bool = True):
        return self.add("insert", a, a_ref, b, b_ref, offset_mm=offset_mm, insert_opposed=opposed)

    def add_angle(self, a, a_ref, b, b_ref, angle_deg: float):
        return self.add("angle", a, a_ref, b, b_ref, angle_deg=angle_deg)


class AssemblyJoints:
    """Inventor-style joint collection over the existing generalized joint backend.

    This is deliberately a façade: all geometry/solver behavior remains in the
    existing CadDocument joint implementation so routing commands through this
    collection cannot create a second source of assembly semantics.
    """

    def __init__(self, parent: "AssemblyComponentDefinition"):
        self.parent = parent

    def __len__(self) -> int:
        return len(self.parent.document.joints)

    def __iter__(self):
        return iter(self.parent.document.joints)

    @staticmethod
    def _side(value: ComponentOccurrence | GeometryProxy | None, ref: str | None, intent: dict[str, Any] | None):
        if isinstance(value, GeometryProxy):
            return value.occurrence_name, value.reference_name, value.geometry_intent
        if isinstance(value, ComponentOccurrence):
            return value.name, ref, copy.deepcopy(intent) if isinstance(intent, dict) else None
        return None, ref, copy.deepcopy(intent) if isinstance(intent, dict) else None

    def add(self, type_: str,
            a: ComponentOccurrence | GeometryProxy | None, a_ref: str | None,
            b: ComponentOccurrence | GeometryProxy | None, b_ref: str | None,
            **kwargs):
        a_occ, ar, ai = self._side(a, a_ref, kwargs.pop("a_intent", None))
        b_occ, br, bi = self._side(b, b_ref, kwargs.pop("b_intent", None))
        linear_position_mm = kwargs.pop("linear_position_mm", None)
        linear_start_mm = kwargs.pop("linear_start_mm", None)
        linear_end_mm = kwargs.pop("linear_end_mm", None)
        angular_position_deg = kwargs.pop("angular_position_deg", None)
        angular_start_deg = kwargs.pop("angular_start_deg", None)
        angular_end_deg = kwargs.pop("angular_end_deg", None)
        name = kwargs.pop("name", None)
        flip_origin_direction = kwargs.pop("flip_origin_direction", False)
        flip_alignment_direction = kwargs.pop("flip_alignment_direction", False)
        if kwargs:
            raise ValueError(f"Unsupported joint option(s): {sorted(kwargs)}")
        return self.parent.document.add_assembly_joint(
            type_, a_occ, ar, b_occ, br,
            linear_position_mm, linear_start_mm, linear_end_mm,
            angular_position_deg, angular_start_deg, angular_end_deg,
            name, ai, bi, flip_origin_direction, flip_alignment_direction,
        )

    def edit(self, name: str, **updates):
        return self.parent.document.edit_assembly_joint(name, **updates)

    def set_limits(self, name: str, linear_start_mm=None, linear_end_mm=None,
                   angular_start_deg=None, angular_end_deg=None):
        return self.parent.document.set_joint_limits(
            name, linear_start_mm, linear_end_mm, angular_start_deg, angular_end_deg
        )

    def drive(self, name: str, linear_position_mm=None, angular_position_deg=None):
        return self.parent.document.drive_joint(name, linear_position_mm, angular_position_deg)

    def list(self):
        return self.parent.document.list_joints()

    def delete(self, name: str):
        return self.parent.document.delete_joint(name)


class ReferenceManager:
    """Small persistent-reference façade inspired by Inventor ReferenceKeyManager.

    Keys are StandaloneCAD keys, not Autodesk byte-for-byte reference keys.  They are
    stable across save/reload as long as the occurrence name and referenced local
    geometry descriptor remain valid.  Actual topology rebinding is delegated to the
    existing persistent-topology/GeometryIntent backend.
    """

    VERSION = 1

    def __init__(self, document):
        self.document = document

    def create_geometry_proxy(self, occurrence: ComponentOccurrence | str, geometry: str | dict[str, Any]) -> GeometryProxy:
        if isinstance(occurrence, str):
            occurrence = self.document.component_definition.occurrences.item(occurrence)
        if not isinstance(occurrence, ComponentOccurrence):
            raise TypeError("occurrence must be ComponentOccurrence or occurrence name")
        if occurrence.parent.document is not self.document:
            raise ValueError("Occurrence belongs to a different document")
        if not isinstance(geometry, (str, dict)):
            raise TypeError("geometry must be a named reference string or GeometryIntent dict")
        proxy = GeometryProxy(occurrence, copy.deepcopy(geometry))
        # Resolve once now so invalid references fail at object creation instead of much later.
        proxy.world_frame()
        return proxy

    def get_reference_key(self, obj: ComponentOccurrence | GeometryProxy) -> str:
        if isinstance(obj, ComponentOccurrence):
            payload = {
                "v": self.VERSION,
                "kind": "occurrence",
                "name": obj.name,
                "path": str(Path(obj.path).expanduser().resolve()),
            }
            return _encode_key(payload)
        if isinstance(obj, GeometryProxy):
            payload = {
                "v": self.VERSION,
                "kind": "geometry_proxy",
                "occurrence": obj.occurrence.name,
                "path": str(Path(obj.occurrence.path).expanduser().resolve()),
                "entity": copy.deepcopy(obj.entity),
            }
            return _encode_key(payload)
        raise TypeError("ReferenceManager supports ComponentOccurrence and GeometryProxy")

    def can_bind_key_to_object(self, key: str) -> bool:
        try:
            self.bind_key_to_object(key)
            return True
        except Exception:
            return False

    def bind_key_to_object(self, key: str) -> ComponentOccurrence | GeometryProxy:
        payload = _decode_key(key)
        if int(payload.get("v", 0)) != self.VERSION:
            raise ValueError("Unsupported standalone reference-key version")
        kind = payload.get("kind")
        comp = self.document.component_definition
        if not isinstance(comp, AssemblyComponentDefinition):
            raise ValueError("Occurrence/proxy reference keys require an assembly document")
        name = str(payload.get("name") if kind == "occurrence" else payload.get("occurrence"))
        occ = comp.occurrences.item(name)
        expected = str(payload.get("path") or "")
        if expected and str(Path(occ.path).expanduser().resolve()) != expected:
            raise ValueError(f"Occurrence path mismatch while binding key: {name}")
        if kind == "occurrence":
            return occ
        if kind == "geometry_proxy":
            return self.create_geometry_proxy(occ, payload.get("entity"))
        raise ValueError(f"Unsupported reference-key kind: {kind}")



# Autodesk Inventor-compatible enum values used by the additive feature façade.
# These are API-layer values only; the canonical core continues to use its existing
# string operations/directions and the ipt-mcp/tool/planner surface is untouched.
kJoinOperation = 20481
kCutOperation = 20482
kIntersectOperation = 20483
kSurfaceOperation = 20484
kNewBodyOperation = 20485

kPositiveExtentDirection = 20993
kNegativeExtentDirection = 20994
kSymmetricExtentDirection = 20995


_OPERATION_TO_BACKEND = {
    kJoinOperation: "join",
    kCutOperation: "cut",
    kIntersectOperation: "intersect",
    kNewBodyOperation: "new",
}
_OPERATION_ALIASES = {
    "kjoinoperation": "join",
    "join": "join",
    "add": "join",
    "kcutoperation": "cut",
    "cut": "cut",
    "subtract": "cut",
    "kintersectoperation": "intersect",
    "intersect": "intersect",
    "common": "intersect",
    "knewbodyoperation": "new",
    "newbody": "new",
    "new": "new",
}
_BACKEND_TO_OPERATION = {
    "join": kJoinOperation,
    "add": kJoinOperation,
    "cut": kCutOperation,
    "subtract": kCutOperation,
    "intersect": kIntersectOperation,
    "common": kIntersectOperation,
    "new": kNewBodyOperation,
}
_DIRECTION_TO_BACKEND = {
    kPositiveExtentDirection: "positive",
    kNegativeExtentDirection: "negative",
    kSymmetricExtentDirection: "symmetric",
}
_DIRECTION_ALIASES = {
    "kpositiveextentdirection": "positive",
    "positive": "positive",
    "knegativeextentdirection": "negative",
    "negative": "negative",
    "ksymmetricextentdirection": "symmetric",
    "symmetric": "symmetric",
}
_BACKEND_TO_DIRECTION = {
    "positive": kPositiveExtentDirection,
    "negative": kNegativeExtentDirection,
    "symmetric": kSymmetricExtentDirection,
}


def _inventor_operation_to_backend(value: Any) -> str:
    if value == kSurfaceOperation or str(value).strip().lower() == "ksurfaceoperation":
        raise NotImplementedError("kSurfaceOperation requires surface-body support and is not emulated by this façade")
    if isinstance(value, int):
        try:
            return _OPERATION_TO_BACKEND[value]
        except KeyError as exc:
            raise ValueError(f"Unsupported PartFeatureOperationEnum value: {value}") from exc
    key = str(value).strip().lower().replace("_", "")
    try:
        return _OPERATION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported extrude operation: {value}") from exc


def _inventor_direction_to_backend(value: Any) -> str:
    if isinstance(value, int):
        try:
            return _DIRECTION_TO_BACKEND[value]
        except KeyError as exc:
            raise ValueError(f"Unsupported PartFeatureExtentDirectionEnum value: {value}") from exc
    key = str(value).strip().lower().replace("_", "")
    try:
        return _DIRECTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported extrude extent direction: {value}") from exc


def _inventor_length_to_mm(document, value: Any) -> float:
    """Inventor Variant length semantics at the façade boundary.

    Inventor numeric API values use database length units (centimeters), while a
    string can carry explicit units or falls back to document units.  StandaloneCAD's
    canonical core remains millimeter based, so conversion happens only here.
    """
    if isinstance(value, bool):
        raise TypeError("Distance must be a numeric value or expression string")
    if isinstance(value, (int, float)):
        mm = float(value) * 10.0
    elif isinstance(value, str):
        mm = float(document.value(value))
    else:
        raise TypeError("Distance must be a numeric value or expression string")
    if mm <= 0:
        raise ValueError("Extrude distance must be > 0")
    return mm


class Profile:
    """Minimal Inventor Profile bridge for the existing sketch-backed core.

    StandaloneCAD currently extrudes an entire closed sketch.  This wrapper therefore
    identifies that sketch without pretending to implement Inventor's per-loop
    ProfilePath editing yet.  It is sufficient for the real ExtrudeDefinition object
    flow while leaving the existing sketch and planner contracts untouched.
    """

    def __init__(self, document, sketch_name: str):
        name = str(sketch_name)
        if name not in document.sketches:
            raise KeyError(f"Sketch not found: {name}")
        self.document = document
        self.sketch_name = name

    @property
    def Parent(self):
        return self.document.sketches[self.sketch_name]

    @property
    def parent(self):
        return self.Parent


def _profile_sketch_name(document, profile: Any) -> str:
    if isinstance(profile, Profile):
        if profile.document is not document:
            raise ValueError("Profile belongs to a different document")
        return profile.sketch_name
    if isinstance(profile, str):
        name = profile
    elif isinstance(profile, dict) and profile.get("sketch_name"):
        name = str(profile["sketch_name"])
    elif getattr(profile, "sketch_name", None):
        name = str(profile.sketch_name)
    elif getattr(profile, "name", None) and str(profile.name) in document.sketches:
        # Accept the native SketchModel as a compatibility bridge.
        name = str(profile.name)
    else:
        raise TypeError("Profile must be a Profile, sketch name, SketchModel, or {'sketch_name': ...}")
    if name not in document.sketches:
        raise KeyError(f"Sketch not found: {name}")
    return name


class ExtrudeDefinition:
    """Inventor-style mutable definition object backed by the existing parametric core."""

    def __init__(self, document, profile: Any, operation: Any,
                 distance_mm: float = 25.4,
                 extent_direction: Any = kPositiveExtentDirection):
        self.document = document
        self._profile = Profile(document, _profile_sketch_name(document, profile))
        self._operation = _inventor_operation_to_backend(operation)
        self._distance_mm = float(distance_mm)
        self._direction = _inventor_direction_to_backend(extent_direction)
        if self._distance_mm <= 0:
            raise ValueError("Extrude distance must be > 0")

    @classmethod
    def from_feature(cls, document, feature):
        distance_key = "distance_mm" if "distance_mm" in feature.params else "distance"
        distance_mm = float(document.value(feature.params.get(distance_key, 10.0)))
        return cls(
            document,
            Profile(document, feature.params["sketch_name"]),
            _BACKEND_TO_OPERATION.get(str(feature.operation).lower(), kJoinOperation),
            distance_mm=distance_mm,
            extent_direction=_BACKEND_TO_DIRECTION.get(
                str(feature.params.get("direction", "positive")).lower(),
                kPositiveExtentDirection,
            ),
        )

    def Copy(self) -> "ExtrudeDefinition":
        return ExtrudeDefinition(
            self.document,
            Profile(self.document, self._profile.sketch_name),
            _BACKEND_TO_OPERATION[self._operation],
            distance_mm=self._distance_mm,
            extent_direction=_BACKEND_TO_DIRECTION[self._direction],
        )

    def copy(self) -> "ExtrudeDefinition":
        return self.Copy()

    def SetDistanceExtent(self, Distance: Any, ExtentDirection: Any) -> None:
        self._distance_mm = _inventor_length_to_mm(self.document, Distance)
        self._direction = _inventor_direction_to_backend(ExtentDirection)

    def set_distance_extent(self, distance: Any, extent_direction: Any) -> None:
        self.SetDistanceExtent(distance, extent_direction)

    @property
    def Profile(self) -> Profile:
        return self._profile

    @Profile.setter
    def Profile(self, value: Any) -> None:
        self._profile = Profile(self.document, _profile_sketch_name(self.document, value))

    @property
    def profile(self) -> Profile:
        return self.Profile

    @profile.setter
    def profile(self, value: Any) -> None:
        self.Profile = value

    @property
    def Operation(self) -> int:
        return _BACKEND_TO_OPERATION[self._operation]

    @Operation.setter
    def Operation(self, value: Any) -> None:
        self._operation = _inventor_operation_to_backend(value)

    @property
    def operation(self) -> int:
        return self.Operation

    @operation.setter
    def operation(self, value: Any) -> None:
        self.Operation = value

    @property
    def IsTwoDirectional(self) -> bool:
        return False

    @property
    def distance_mm(self) -> float:
        return float(self._distance_mm)

    @property
    def extent_direction(self) -> int:
        return _BACKEND_TO_DIRECTION[self._direction]


class ExtrudeFeature:
    """Inventor-style object wrapper around one native extrude FeatureRecord."""

    def __init__(self, parent: "ExtrudeFeatures", native):
        if native.kind != "extrude":
            raise TypeError(f"ExtrudeFeature requires an extrude record, got {native.kind}")
        self.parent = parent
        self.document = parent.document
        self._native = native

    @property
    def Name(self) -> str:
        return self._native.name

    @Name.setter
    def Name(self, value: str) -> None:
        self.document.edit_feature_definition(self._native.name, {}, new_name=str(value))
        self.document.mark_dirty(True)

    @property
    def name(self) -> str:
        return self.Name

    @name.setter
    def name(self, value: str) -> None:
        self.Name = value

    @property
    def Suppressed(self) -> bool:
        return bool(self._native.suppressed)

    @Suppressed.setter
    def Suppressed(self, value: bool) -> None:
        idx = self.document.features.index(self._native)
        self._native.suppressed = bool(value)
        self.document.rebuild(idx)
        self.document.mark_dirty(True)

    @property
    def suppressed(self) -> bool:
        return self.Suppressed

    @suppressed.setter
    def suppressed(self, value: bool) -> None:
        self.Suppressed = value

    @property
    def Definition(self) -> ExtrudeDefinition:
        return ExtrudeDefinition.from_feature(self.document, self._native)

    @Definition.setter
    def Definition(self, value: ExtrudeDefinition) -> None:
        if not isinstance(value, ExtrudeDefinition):
            raise TypeError("Definition must be an ExtrudeDefinition")
        if value.document is not self.document:
            raise ValueError("ExtrudeDefinition belongs to a different document")
        current_sketch = str(self._native.params.get("sketch_name"))
        if value.Profile.sketch_name != current_sketch:
            raise NotImplementedError(
                "Changing an existing ExtrudeFeature.Profile is deferred until full Inventor Profiles semantics are added"
            )
        self.document.edit_feature_definition(
            self._native.name,
            {
                "distance_mm": value.distance_mm,
                "direction": value._direction,
                "operation": value._operation,
            },
        )
        self.document.mark_dirty(True)

    @property
    def definition(self) -> ExtrudeDefinition:
        return self.Definition

    @definition.setter
    def definition(self, value: ExtrudeDefinition) -> None:
        self.Definition = value

    @property
    def Parent(self) -> "ExtrudeFeatures":
        return self.parent

    def Delete(self) -> None:
        self.document.delete_feature(self._native.name)
        self.document.mark_dirty(True)

    def delete(self) -> None:
        self.Delete()


class ExtrudeFeatures:
    """Inventor ExtrudeFeatures-compatible collection over native history records."""

    def __init__(self, parent: "PartFeatures"):
        self.parent = parent
        self.document = parent.document

    def _records(self):
        return [f for f in self.document.features if f.kind == "extrude"]

    @property
    def Count(self) -> int:
        return len(self._records())

    @property
    def count(self) -> int:
        return self.Count

    def Item(self, key: int | str) -> ExtrudeFeature:
        records = self._records()
        if isinstance(key, int):
            if key < 1:
                raise IndexError("ExtrudeFeatures index is 1-based")
            try:
                return ExtrudeFeature(self, records[key - 1])
            except IndexError as exc:
                raise IndexError(f"ExtrudeFeatures index out of range: {key}") from exc
        name = str(key)
        native = next((f for f in records if f.name == name), None)
        if native is None:
            raise KeyError(f"ExtrudeFeature not found: {name}")
        return ExtrudeFeature(self, native)

    def item(self, key: int | str) -> ExtrudeFeature:
        return self.Item(key)

    def __len__(self) -> int:
        return self.Count

    def __iter__(self) -> Iterator[ExtrudeFeature]:
        for native in self._records():
            yield ExtrudeFeature(self, native)

    def __getitem__(self, key: int | str) -> ExtrudeFeature:
        return self.Item(key)

    def CreateExtrudeDefinition(self, Profile: Any, Operation: Any) -> ExtrudeDefinition:
        # Inventor initializes a new ExtrudeDefinition with a 1.0 in distance extent.
        return ExtrudeDefinition(self.document, Profile, Operation, distance_mm=25.4)

    def create_extrude_definition(self, profile: Any, operation: Any) -> ExtrudeDefinition:
        return self.CreateExtrudeDefinition(profile, operation)

    def Add(self, Definition: ExtrudeDefinition) -> ExtrudeFeature:
        if not isinstance(Definition, ExtrudeDefinition):
            raise TypeError("Definition must be an ExtrudeDefinition")
        if Definition.document is not self.document:
            raise ValueError("ExtrudeDefinition belongs to a different document")
        if Definition._operation == "new" and self.document.shape is not None:
            raise NotImplementedError(
                "Inventor kNewBodyOperation on an existing body requires true multi-body support; refusing to fake it by unioning bodies"
            )
        distance_param = self.document.create_model_parameter(Definition.distance_mm, "mm")
        rec = self.document.add_feature(
            "extrude",
            None,
            {
                "sketch_name": Definition.Profile.sketch_name,
                "distance_mm": distance_param,
                "direction": Definition._direction,
            },
            Definition._operation,
        )
        self.document.mark_dirty(True)
        return ExtrudeFeature(self, rec)

    def add(self, definition: ExtrudeDefinition) -> ExtrudeFeature:
        return self.Add(definition)


class EdgeLoop:
    """Minimal Inventor-style EdgeLoop reference usable as a Loft section.

    The wrapper stores the owning face as a persistent reference plus either its outer
    loop or a 1-based inner loop index.  It deliberately does not pretend to expose the
    complete Inventor EdgeLoop/EdgeUse object model.
    """

    def __init__(self, document, face_ref: Any, loop: str | int = "outer"):
        if document.shape is None:
            raise ValueError("EdgeLoop requires existing part geometry")
        self.document = document
        self.face_ref = make_face_ref(document.shape, face_ref)
        if str(loop).lower() == "outer":
            self.loop: str | int = "outer"
        else:
            index = int(loop)
            if index < 1:
                raise ValueError("Inner EdgeLoop index is 1-based")
            self.loop = index

    def as_section(self) -> dict[str, Any]:
        return {"type": "edge_loop", "face_ref": copy.deepcopy(self.face_ref), "loop": self.loop}


class LoftDefinition:
    """Inventor LoftDefinition-compatible subset over CADia's parametric loft core.

    Supported now: ordered closed sections, PartFeatureOperationEnum, Closed=False and
    MergeTangentFaces.  Rails/centerline/boundary-condition/map-curve controls are not
    silently approximated; callers that need them should use Inventor or wait for a
    kernel implementation that can represent those semantics exactly.
    """

    def __init__(self, document, sections: list[Any], operation: Any):
        self.document = document
        self._sections = [self._coerce_section(x) for x in list(sections or [])]
        if len(self._sections) < 2:
            raise ValueError("LoftDefinition requires at least two sections")
        self._operation = _inventor_operation_to_backend(operation)
        self._closed = False
        self._merge_tangent_faces = True

    def _coerce_section(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Profile):
            if value.document is not self.document:
                raise ValueError("Profile belongs to a different document")
            return {"type": "sketch", "sketch_name": value.sketch_name}
        if isinstance(value, EdgeLoop):
            if value.document is not self.document:
                raise ValueError("EdgeLoop belongs to a different document")
            return value.as_section()
        if isinstance(value, str) and value in self.document.sketches:
            return {"type": "sketch", "sketch_name": value}
        if isinstance(value, dict):
            kind = str(value.get("type") or "").lower()
            if kind == "sketch" and value.get("sketch_name"):
                name = str(value["sketch_name"])
                if name not in self.document.sketches:
                    raise KeyError(f"Sketch not found: {name}")
                return {"type": "sketch", "sketch_name": name}
            if kind in ("face", "edge_loop") and value.get("face_ref"):
                if self.document.shape is None:
                    raise ValueError(f"{kind} section requires existing geometry")
                out = {"type": kind, "face_ref": make_face_ref(self.document.shape, value["face_ref"])}
                if kind == "edge_loop":
                    out["loop"] = value.get("loop", "outer")
                return out
        raise TypeError("Loft section must be a Profile, EdgeLoop, sketch name, or section descriptor")

    def Copy(self) -> "LoftDefinition":
        dup = object.__new__(LoftDefinition)
        dup.document = self.document
        dup._sections = copy.deepcopy(self._sections)
        dup._operation = self._operation
        dup._closed = self._closed
        dup._merge_tangent_faces = self._merge_tangent_faces
        return dup

    def copy(self) -> "LoftDefinition":
        return self.Copy()

    @property
    def Sections(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._sections)

    @property
    def sections(self) -> list[dict[str, Any]]:
        return self.Sections

    @property
    def Operation(self) -> int:
        return _BACKEND_TO_OPERATION[self._operation]

    @Operation.setter
    def Operation(self, value: Any) -> None:
        self._operation = _inventor_operation_to_backend(value)

    @property
    def operation(self) -> int:
        return self.Operation

    @operation.setter
    def operation(self, value: Any) -> None:
        self.Operation = value

    @property
    def Closed(self) -> bool:
        return self._closed

    @Closed.setter
    def Closed(self, value: bool) -> None:
        self._closed = bool(value)

    @property
    def MergeTangentFaces(self) -> bool:
        return self._merge_tangent_faces

    @MergeTangentFaces.setter
    def MergeTangentFaces(self, value: bool) -> None:
        self._merge_tangent_faces = bool(value)


class LoftFeature:
    def __init__(self, parent: "LoftFeatures", native):
        if native.kind != "loft":
            raise TypeError(f"LoftFeature requires a loft record, got {native.kind}")
        self.parent = parent
        self.document = parent.document
        self._native = native

    @property
    def Name(self) -> str:
        return self._native.name

    @Name.setter
    def Name(self, value: str) -> None:
        self.document.edit_feature_definition(self._native.name, {}, new_name=str(value))
        self.document.mark_dirty(True)

    @property
    def Definition(self) -> LoftDefinition:
        # Stored face/EdgeLoop sections describe the pre-loft topology and may be
        # consumed by the resulting feature.  Return the persisted definition directly
        # instead of trying to rebind it against post-feature geometry.
        sections = self._native.params.get("sections") or [
            {"type": "sketch", "sketch_name": n} for n in (self._native.params.get("sketch_names") or [])
        ]
        obj = object.__new__(LoftDefinition)
        obj.document = self.document
        obj._sections = copy.deepcopy(sections)
        obj._operation = str(self._native.operation).lower()
        obj._closed = False
        obj._merge_tangent_faces = bool(self._native.params.get("merge_tangent_faces", True))
        return obj

    @Definition.setter
    def Definition(self, value: LoftDefinition) -> None:
        if not isinstance(value, LoftDefinition):
            raise TypeError("Definition must be a LoftDefinition")
        if value.document is not self.document:
            raise ValueError("LoftDefinition belongs to a different document")
        if value.Closed:
            raise NotImplementedError("Inventor LoftDefinition.Closed=True is not implemented by the current solid kernel")
        self.document.edit_feature_definition(
            self._native.name,
            {"sections": value.Sections, "merge_tangent_faces": value.MergeTangentFaces, "operation": value._operation},
        )
        self.document.mark_dirty(True)

    def Delete(self) -> None:
        self.document.delete_feature(self._native.name)
        self.document.mark_dirty(True)


class LoftFeatures:
    """Inventor LoftFeatures collection with CreateLoftDefinition/Add semantics."""

    def __init__(self, parent: "PartFeatures"):
        self.parent = parent
        self.document = parent.document

    def _records(self):
        return [f for f in self.document.features if f.kind == "loft"]

    @property
    def Count(self) -> int:
        return len(self._records())

    def Item(self, key: int | str) -> LoftFeature:
        records = self._records()
        if isinstance(key, int):
            if key < 1:
                raise IndexError("LoftFeatures index is 1-based")
            try:
                return LoftFeature(self, records[key - 1])
            except IndexError as exc:
                raise IndexError(f"LoftFeatures index out of range: {key}") from exc
        name = str(key)
        native = next((f for f in records if f.name == name), None)
        if native is None:
            raise KeyError(f"LoftFeature not found: {name}")
        return LoftFeature(self, native)

    def CreateLoftDefinition(self, LoftSections: Any, Operation: Any) -> LoftDefinition:
        return LoftDefinition(self.document, list(LoftSections or []), Operation)

    def create_loft_definition(self, loft_sections: Any, operation: Any) -> LoftDefinition:
        return self.CreateLoftDefinition(loft_sections, operation)

    def Add(self, Definition: LoftDefinition) -> LoftFeature:
        if not isinstance(Definition, LoftDefinition):
            raise TypeError("Definition must be a LoftDefinition")
        if Definition.document is not self.document:
            raise ValueError("LoftDefinition belongs to a different document")
        if Definition.Closed:
            raise NotImplementedError("Inventor LoftDefinition.Closed=True is not implemented by the current solid kernel")
        rec = self.document.add_feature(
            "loft",
            None,
            {
                "sections": Definition.Sections,
                "ruled": False,
                "merge_tangent_faces": Definition.MergeTangentFaces,
            },
            Definition._operation,
        )
        self.document.mark_dirty(True)
        return LoftFeature(self, rec)

    def add(self, definition: LoftDefinition) -> LoftFeature:
        return self.Add(definition)


class PartFeatures:
    """Additive Inventor-style typed feature collection.

    The existing ``PartComponentDefinition.features`` property deliberately remains the
    raw native history list for backward compatibility.  Inventor-style callers use the
    new PascalCase ``PartComponentDefinition.Features`` property instead.
    """

    def __init__(self, document):
        self.document = document
        self._extrude_features = ExtrudeFeatures(self)
        self._loft_features = LoftFeatures(self)

    @property
    def ExtrudeFeatures(self) -> ExtrudeFeatures:
        return self._extrude_features

    @property
    def extrude_features(self) -> ExtrudeFeatures:
        return self._extrude_features

    @property
    def LoftFeatures(self) -> LoftFeatures:
        return self._loft_features

    @property
    def loft_features(self) -> LoftFeatures:
        return self._loft_features

    @property
    def Count(self) -> int:
        return len(self.document.features)

    def Item(self, key: int | str):
        if isinstance(key, int):
            if key < 1:
                raise IndexError("PartFeatures index is 1-based")
            try:
                native = self.document.features[key - 1]
            except IndexError as exc:
                raise IndexError(f"PartFeatures index out of range: {key}") from exc
        else:
            name = str(key)
            native = next((f for f in self.document.features if f.name == name), None)
            if native is None:
                raise KeyError(f"Feature not found: {name}")
        if native.kind == "extrude":
            return ExtrudeFeature(self._extrude_features, native)
        if native.kind == "loft":
            return LoftFeature(self._loft_features, native)
        return native

    def item(self, key: int | str):
        return self.Item(key)

class ComponentDefinition:
    def __init__(self, document):
        self.document = document

    @property
    def reference_manager(self) -> ReferenceManager:
        return self.document.reference_manager

    @property
    def type(self) -> str:
        return self.document.doc_type


class PartComponentDefinition(ComponentDefinition):
    def __init__(self, document):
        super().__init__(document)
        self._inventor_features = PartFeatures(document)

    @property
    def Features(self) -> PartFeatures:
        return self._inventor_features

    @property
    def features(self):
        # Backward-compatible raw StandaloneCAD history list.
        return self.document.features

    @property
    def sketches(self):
        return self.document.sketches

    @property
    def parameters(self):
        return self.document.parameters

    @property
    def work_planes(self):
        return self.document.work_planes

    @property
    def work_axes(self):
        return self.document.work_axes

    @property
    def work_points(self):
        return getattr(self.document,'work_points',{})

    @property
    def sketches3d(self):
        return getattr(self.document,'sketches3d',{})

    @property
    def surface_bodies(self):
        return [] if self.document.shape is None else [self.document.shape]


class AssemblyComponentDefinition(ComponentDefinition):
    def __init__(self, document):
        super().__init__(document)
        self._occurrences = ComponentOccurrences(self)
        self._constraints = AssemblyConstraints(self)
        self._joints = AssemblyJoints(self)

    @property
    def occurrences(self) -> ComponentOccurrences:
        return self._occurrences

    @property
    def constraints(self) -> AssemblyConstraints:
        return self._constraints

    @property
    def joints(self) -> AssemblyJoints:
        return self._joints

    @property
    def bom(self) -> dict[str, Any]:
        return self.document.assembly_bom()

    def list_interfaces(self, occurrence: str | None = None):
        return self.document.list_interfaces(occurrence)

    def check_interference(self, occurrences=None):
        return self.document.check_interference(occurrences)

    def measure_min_distance(self, a_occurrence: str, a_ref: str | None,
                             b_occurrence: str, b_ref: str | None):
        return self.document.measure_min_distance(a_occurrence, a_ref, b_occurrence, b_ref)

    def get_bom(self, max_rows: int = 500):
        return self.document.assembly_bom(int(max_rows))

    def list_constraints(self):
        return {'constraints': [vars(c) for c in self.document.constraints]}


def component_definition_for(document) -> ComponentDefinition:
    if document.doc_type == "assembly":
        return AssemblyComponentDefinition(document)
    return PartComponentDefinition(document)
