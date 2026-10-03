from __future__ import annotations

import pytest

from standalonecad.core.document import CadDocument, SketchEntity, SketchModel
from standalonecad.core.engine import CadEngine
from standalonecad.core.inventor_facade import (
    Profile,
    kJoinOperation,
    kPositiveExtentDirection,
    kSymmetricExtentDirection,
)
from standalonecad.mcp.tools import make_tools


def _native_extrude(height_mm=4.0):
    e = CadEngine()
    e.execute('create_sketch', {'plane': 'XY'})
    e.execute('draw_circle', {'cx': 0, 'cy': 0, 'radius': 5})
    e.execute('close_sketch', {'sketch_name': None})
    e.execute('extrude', {
        'sketch_name': 'Sketch1',
        'distance_mm': height_mm,
        'operation': 'join',
        'direction': 'positive',
    })
    return e


def test_inventor_extrude_facade_is_additive_and_does_not_change_tool_surface():
    e = _native_extrude()
    comp = e.doc.component_definition

    # Legacy/raw access is deliberately unchanged.
    assert comp.features is e.doc.features

    # Inventor-style access is additive.
    extrudes = comp.Features.ExtrudeFeatures
    assert extrudes.Count == 1
    assert extrudes.Item(1).Name == 'Extrude1'
    assert comp.Features.Item('Extrude1').Name == 'Extrude1'

    # ipt-mcp contract remains frozen.
    assert len(make_tools('inventor')) == 58


def test_definition_copy_set_distance_extent_and_assign_preserves_existing_d_parameter():
    e = _native_extrude(4.0)
    feature = e.doc.component_definition.Features.ExtrudeFeatures.Item(1)
    param_name = e.doc.features[0].params['distance_mm']

    definition = feature.Definition.Copy()
    # Inventor numeric database length units are centimeters: 2.0 => 20 mm.
    definition.SetDistanceExtent(2.0, kPositiveExtentDirection)
    feature.Definition = definition

    assert e.doc.features[0].params['distance_mm'] == param_name
    assert e.doc.parameters[param_name].expression == '20 mm'
    assert e.doc.shape.BoundingBox().zmax == pytest.approx(20.0)


def test_create_definition_add_uses_inventor_style_flow_without_engine_or_planner_changes():
    d = CadDocument(); d.reset('part')
    sm = SketchModel('Sketch1', 'XY')
    sm.entities.append(SketchEntity('circle', 'E1', {'cx': 0, 'cy': 0, 'r': 5}))
    sm.closed = True
    d.sketches['Sketch1'] = sm

    extrudes = d.component_definition.Features.ExtrudeFeatures
    definition = extrudes.CreateExtrudeDefinition(Profile(d, 'Sketch1'), kJoinOperation)
    definition.SetDistanceExtent('12 mm', kPositiveExtentDirection)
    feature = extrudes.Add(definition)

    assert feature.Name == 'Extrude1'
    assert extrudes.Count == 1
    assert d.shape.BoundingBox().zmax == pytest.approx(12.0)
    assert d.features[0].params['distance_mm'] in d.parameters


def test_symmetric_extent_maps_to_existing_core_semantics():
    d = CadDocument(); d.reset('part')
    sm = SketchModel('Sketch1', 'XY')
    sm.entities.append(SketchEntity('circle', 'E1', {'cx': 0, 'cy': 0, 'r': 5}))
    sm.closed = True
    d.sketches['Sketch1'] = sm

    extrudes = d.component_definition.Features.ExtrudeFeatures
    definition = extrudes.CreateExtrudeDefinition(Profile(d, 'Sketch1'), kJoinOperation)
    definition.SetDistanceExtent(1.0, kSymmetricExtentDirection)  # 10 mm each side in the existing CQ both-direction semantics
    feature = extrudes.Add(definition)

    assert feature.Definition.extent_direction == kSymmetricExtentDirection
    bb = d.shape.BoundingBox()
    assert bb.zmin == pytest.approx(-10.0)
    assert bb.zmax == pytest.approx(10.0)
