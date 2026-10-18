from __future__ import annotations
import hashlib, json, math
from typing import Any
import numpy as np


def _r(v,n=6):
    try:return round(float(v),n)
    except Exception:return 0.0


def _hash(prefix,data):
    raw=json.dumps(data,sort_keys=True,separators=(',',':'))
    return f"{prefix}_{hashlib.sha1(raw.encode()).hexdigest()[:14]}"


def _direction_name(v):
    a=np.asarray(v,float)
    if np.linalg.norm(a)<1e-12:return None
    i=int(np.argmax(np.abs(a))); sign='+' if a[i]>=0 else '-'; return sign+'XYZ'[i]


def _bbox_normalized_point(shape, point, bb=None):
    """Return a dimensionless point in [-1, 1]^3 relative to the current body bbox.

    ``bb`` is accepted so topology indexing can compute the body bounding box once per
    shape instead of once per face/edge.  This is a pure performance optimization; the
    descriptor semantics are unchanged.
    """
    try:
        bb=bb or shape.BoundingBox(); vals=[]
        for value,lo,span in ((point.x,bb.xmin,bb.xlen),(point.y,bb.ymin,bb.ylen),(point.z,bb.zmin,bb.zlen)):
            vals.append(0.0 if abs(span)<1e-12 else 2.0*(float(value)-float(lo))/float(span)-1.0)
        return [_r(x,5) for x in vals]
    except Exception:
        return [0.0,0.0,0.0]


def _shape_token(shape, faces, edges, vertices):
    """Cheap guard against an unexpectedly replaced wrapped shape on the same wrapper."""
    try:h=int(shape.hashCode())
    except Exception:h=id(getattr(shape,'wrapped',shape))
    return (h,len(faces),len(edges),len(vertices))


def _topology_index(shape):
    """Build/cache the expensive body topology index once for a CadQuery shape.

    v9.2 used an edge -> every face -> every face-edge scan.  Threaded fasteners and
    gears can contain thousands of edges, making that O(E*F*Ef) path dominate selection
    and persistent-reference work.  Here each face edge is bucketed by OCCT hashCode and
    verified with ``isSame`` before adjacency is recorded.  Hash collisions therefore do
    not change correctness, while typical work becomes close to O(total face-edges).

    The cache is attached to the immutable-ish CadQuery Shape wrapper.  Boolean/modeling
    operations in this engine return a new wrapper, so a new body gets a new cache.  The
    token check is an additional safety guard.
    """
    if shape is None:return None
    cached=getattr(shape,'_standalonecad_topology_index',None)
    try: current_hash=int(shape.hashCode())
    except Exception: current_hash=id(getattr(shape,'wrapped',shape))
    # CadQuery/OCCT modeling operations return a new Shape wrapper.  If a cache is
    # attached to this wrapper and its OCCT hash is unchanged, no enumeration is needed.
    if isinstance(cached,dict) and cached.get('shape_hash')==current_hash:
        return cached
    faces=list(shape.Faces()); edges=list(shape.Edges()); vertices=list(shape.Vertices())
    token=_shape_token(shape,faces,edges,vertices)
    bb=shape.BoundingBox()
    edge_buckets={}
    for i,e in enumerate(edges):
        try:key=int(e.hashCode())
        except Exception:key=hash(e)
        edge_buckets.setdefault(key,[]).append(i)

    edge_adj=[[] for _ in edges]
    face_edge_geoms=[]
    face_edge_counts=[]
    for f in faces:
        fedges=list(f.Edges()); face_edge_counts.append(len(fedges)); face_edge_geoms.append(sorted(str(e.geomType()) for e in fedges))
        fgeom=str(f.geomType())
        for fe in fedges:
            try:key=int(fe.hashCode())
            except Exception:key=hash(fe)
            found=None
            for i in edge_buckets.get(key,()):
                try:
                    if edges[i].isSame(fe):found=i; break
                except Exception:pass
            # Defensive fallback for an unusual OCCT hash mismatch.  It is intentionally
            # only used for the exceptional case, never as the normal adjacency path.
            if found is None:
                for i,e in enumerate(edges):
                    try:
                        if e.isSame(fe):found=i; break
                    except Exception:pass
            if found is not None:edge_adj[found].append(fgeom)

    index={
        'token':token,'shape_hash':current_hash,'bb':bb,'faces':faces,'edges':edges,'vertices':vertices,
        'edge_adj':edge_adj,'face_edge_geoms':face_edge_geoms,'face_edge_counts':face_edge_counts,
        'face_records':None,'edge_records':None,'vertex_records':None,
    }
    try:shape._standalonecad_topology_index=index
    except Exception:pass
    return index


def face_records(shape):
    if shape is None:return []
    idx=_topology_index(shape)
    if idx.get('face_records') is not None:return idx['face_records']
    faces=idx['faces']; bb=idx['bb']; out=[]
    centers=[f.Center() for f in faces]
    extremes={
        '+X':max((c.x for c in centers),default=0), '-X':min((c.x for c in centers),default=0),
        '+Y':max((c.y for c in centers),default=0), '-Y':min((c.y for c in centers),default=0),
        '+Z':max((c.z for c in centers),default=0), '-Z':min((c.z for c in centers),default=0),
    }
    alias_map={'+Z':['top_face','max_z_face'],'-Z':['bottom_face','min_z_face'], '+X':['max_x_face'], '-X':['min_x_face'], '+Y':['max_y_face'], '-Y':['min_y_face']}
    for i,f in enumerate(faces):
        c=centers[i]; geom=str(f.geomType()); area=float(f.Area())
        edge_geoms=idx['face_edge_geoms'][i]
        rec={'index':i,'geom':geom,'center':[_r(c.x),_r(c.y),_r(c.z)],'center_norm':_bbox_normalized_point(shape,c,bb),'area':_r(area,5),'edge_count':idx['face_edge_counts'][i],'edge_geom_signature':edge_geoms}
        try:
            n=f.normalAt(c); nv=[_r(n.x),_r(n.y),_r(n.z)]; rec['normal']=nv; dn=_direction_name(nv)
        except Exception: rec['normal']=[0,0,0]; dn=None
        rec['direction']=dn
        if geom=='CYLINDER':
            try:
                surf=f._geomAdaptor(); cyl=surf.Cylinder(); rec['radius_mm']=_r(cyl.Radius(),6)
                ax=cyl.Axis().Direction(); rec['axis']=[_r(ax.X()),_r(ax.Y()),_r(ax.Z())]; rec['axis_direction']=_direction_name(rec['axis'])
            except Exception: pass
        semantics=[]
        if geom=='PLANE' and dn in extremes:
            coord={'X':c.x,'Y':c.y,'Z':c.z}[dn[1]]
            if abs(coord-extremes[dn])<1e-5: semantics=list(alias_map.get(dn,[]))
        rec['semantic']=semantics[0] if semantics else None
        rec['semantics']=semantics
        sig={k:rec[k] for k in ('geom','center','area','normal','edge_count','edge_geom_signature')}
        if 'radius_mm' in rec:sig['radius_mm']=rec['radius_mm']; sig['axis']=rec.get('axis')
        rec['id']=_hash('F',sig)
        out.append(rec)
    idx['face_records']=out
    return out


def edge_records(shape):
    if shape is None:return []
    idx=_topology_index(shape)
    if idx.get('edge_records') is not None:return idx['edge_records']
    out=[]; bb=idx['bb']
    for i,e in enumerate(idx['edges']):
        c=e.Center(); geom=str(e.geomType()); adj=sorted(idx['edge_adj'][i])
        rec={'index':i,'geom':geom,'center':[_r(c.x),_r(c.y),_r(c.z)],'center_norm':_bbox_normalized_point(shape,c,bb),'length':_r(e.Length(),5),'adjacent_face_geoms':adj,'adjacent_face_count':len(adj)}
        verts=e.Vertices(); rec['vertices']=[[_r(v.Center().x),_r(v.Center().y),_r(v.Center().z)] for v in verts]
        if geom=='LINE' and len(verts)>=2:
            a=np.asarray(rec['vertices'][0],float); b=np.asarray(rec['vertices'][-1],float); rec['direction']=_direction_name(b-a)
        else: rec['direction']=None
        try:
            if geom=='CIRCLE': rec['radius_mm']=_r(e.radius(),6)
        except Exception: pass
        sig={k:rec[k] for k in ('geom','center','length','vertices','adjacent_face_geoms','adjacent_face_count')}; rec['id']=_hash('E',sig); out.append(rec)
    idx['edge_records']=out
    return out


def vertex_records(shape):
    if shape is None:return []
    idx=_topology_index(shape)
    if idx.get('vertex_records') is not None:return idx['vertex_records']
    out=[]
    for i,v in enumerate(idx['vertices']):
        c=v.Center(); sig={'point':[_r(c.x),_r(c.y),_r(c.z)]}; out.append({'id':_hash('V',sig),'index':i,**sig})
    idx['vertex_records']=out
    return out


def _ref_semantics(ref):
    if isinstance(ref,str): return [ref]
    if not isinstance(ref,dict): return []
    vals=[]
    if ref.get('semantic'): vals.append(str(ref['semantic']))
    vals.extend(str(x) for x in (ref.get('semantics') or []))
    return vals




def _bbox_overlap_possible(a, b, tol=1e-7):
    """Cheap, geometry-agnostic rejection for subshapes that cannot overlap."""
    try:
        aa=a.BoundingBox(); bb=b.BoundingBox()
        return not (
            aa.xmax < bb.xmin-tol or bb.xmax < aa.xmin-tol or
            aa.ymax < bb.ymin-tol or bb.ymax < aa.ymin-tol or
            aa.zmax < bb.zmin-tol or bb.zmax < aa.zmin-tol
        )
    except Exception:
        return True


def _subshape_measure(item, kind):
    try:
        return max(0.0, float(item.Area() if kind == 'face' else item.Length()))
    except Exception:
        return 0.0


def _positive_overlap(a, b, kind):
    """Return same-dimensional geometric overlap between two subshapes.

    This deliberately contains no feature-name or geometry-type special cases.
    Face lineage is propagated only through shared *area* and edge lineage only
    through shared *length*, so mere touching at an edge/vertex does not create
    ancestry.  Exact OCCT identity is handled before this function is called.
    """
    if not _bbox_overlap_possible(a, b):
        return 0.0
    try:
        import cadquery as cq
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        result=BRepAlgoAPI_Common(a.wrapped,b.wrapped).Shape()
        common=cq.Shape.cast(result)
        if common is None or common.isNull():
            return 0.0
        value=float(common.Area() if kind == 'face' else common.Length())
        if not math.isfinite(value):
            return 0.0
        return max(0.0,value)
    except Exception:
        return 0.0


def _feature_tag(name, kind):
    return (str(name), str(kind))


def _lineage_owners(shape, kind):
    """Build generic feature ancestry for final faces/edges from stage snapshots.

    The algorithm is feature-agnostic: across each adjacent feature stage it
    first preserves exact OCCT subshape identity, then propagates ancestry across
    same-dimensional geometric overlap.  New topology that has no predecessor
    is assigned to the feature that introduced it.  Splits naturally become
    one-to-many, and merges carry the union of predecessor owners.

    This is intentionally a runtime reference layer; it does not alter B-Rep
    geometry or replace OCCT/CadQuery modeling operations.
    """
    stages=getattr(shape,'_cadia_feature_stages',None)
    if not stages:
        return None
    cache_name='_cadia_generic_lineage_'+kind
    cached=getattr(shape,cache_name,None)
    if cached is not None:
        return cached
    # General performance guard.  If lineage would require an excessive pairwise
    # history walk, retain the pre-existing geometric matcher instead of changing
    # modeling behavior or introducing feature-specific shortcuts.
    if len(stages)>160:
        return None
    getter=(lambda sh:list(sh.Faces())) if kind=='face' else (lambda sh:list(sh.Edges()))
    prev_shapes=[]; prev_owners=[]; current_owners=[]
    for feature_name, feature_kind, stage in stages:
        if stage is None:
            prev_shapes=[]; prev_owners=[]; current_owners=[]
            continue
        cur_shapes=getter(stage)
        if len(cur_shapes)>2200:
            return None
        tag=frozenset((_feature_tag(feature_name,feature_kind),))
        if not prev_shapes:
            current_owners=[tag for _ in cur_shapes]
            prev_shapes=cur_shapes; prev_owners=current_owners
            continue
        current_owners=[None]*len(cur_shapes)
        # First preserve exact OCCT identity.  This is both fastest and strongest.
        for j,new_item in enumerate(cur_shapes):
            for i,old_item in enumerate(prev_shapes):
                try:same=bool(old_item.isSame(new_item))
                except Exception:same=False
                if same:
                    current_owners[j]=prev_owners[i]
                    break
        # Then propagate ancestry through genuine same-dimensional overlap.
        # A relative epsilon suppresses numerical slivers from boolean tolerances.
        for j,new_item in enumerate(cur_shapes):
            if current_owners[j] is not None:
                continue
            new_measure=_subshape_measure(new_item,kind)
            ancestors=set()
            for i,old_item in enumerate(prev_shapes):
                old_measure=_subshape_measure(old_item,kind)
                scale=max(1.0,min(x for x in (new_measure,old_measure) if x>0) if (new_measure>0 and old_measure>0) else 1.0)
                overlap=_positive_overlap(old_item,new_item,kind)
                if overlap>max(1e-8,scale*1e-7):
                    ancestors.update(prev_owners[i])
            current_owners[j]=frozenset(ancestors) if ancestors else tag
        prev_shapes=cur_shapes; prev_owners=current_owners
    if not stages or stages[-1][2] is not shape:
        return None
    try:setattr(shape,cache_name,current_owners)
    except Exception:pass
    return current_owners


def _lineage_payload(shape, kind, index):
    owners=_lineage_owners(shape,kind)
    if owners is None or index<0 or index>=len(owners) or not owners[index]:
        return None
    return [{'feature_name':name,'feature_kind':ftype} for name,ftype in sorted(owners[index])]


def _lineage_allowed(shape, kind, ref):
    """Return indices carrying any persisted ancestor, or None for legacy refs."""
    if not isinstance(ref,dict) or not ref.get('lineage_owners'):
        return None
    expected={_feature_tag(x.get('feature_name'),x.get('feature_kind')) for x in ref.get('lineage_owners',[]) if isinstance(x,dict)}
    if not expected:
        return None
    owners=_lineage_owners(shape,kind)
    if owners is None:
        # Do not change behavior for environments where runtime ancestry cannot be
        # computed.  The legacy matcher remains available for old/imported models.
        return None
    allowed={i for i,tags in enumerate(owners) if tags and expected.intersection(tags)}
    if not allowed:
        raise ValueError('Referenced topology lineage no longer exists; reselect the topology')
    return allowed

def make_face_ref(shape, ref):
    """Capture a persistent descriptor for a current face id/semantic/selector."""
    face=match_face(shape,ref)
    idx=_topology_index(shape)
    for r,f in zip(face_records(shape),idx['faces']):
        if f.isSame(face):
            out={k:r.get(k) for k in ('id','semantic','semantics','geom','direction','center','center_norm','area','normal','radius_mm','axis_direction','edge_count','edge_geom_signature') if r.get(k) is not None}
            lineage=_lineage_payload(shape,'face',r['index'])
            if lineage: out['lineage_owners']=lineage
            return out
    raise ValueError(f'Face reference not found: {ref}')


def make_edge_ref(shape, ref):
    edge=match_edge(shape,ref)
    idx=_topology_index(shape)
    for r,e in zip(edge_records(shape),idx['edges']):
        if e.isSame(edge):
            out={k:r.get(k) for k in ('id','geom','direction','center','center_norm','length','vertices','radius_mm','adjacent_face_geoms','adjacent_face_count') if r.get(k) is not None}
            lineage=_lineage_payload(shape,'edge',r['index'])
            if lineage: out['lineage_owners']=lineage
            return out
    raise ValueError(f'Edge reference not found: {ref}')


def make_vertex_ref(shape, ref):
    """Capture a persistent descriptor for a current vertex id.

    The exact id remains the primary lookup key.  ``point_norm`` is only used as a
    fallback after upstream topology changes, so already-valid vertex references keep
    their legacy path and cost.
    """
    v=match_vertex_id(shape,ref)
    bb=shape.BoundingBox(); c=v.Center()
    point=[_r(c.x),_r(c.y),_r(c.z)]
    return {
        'id': next((r['id'] for r,vv in zip(vertex_records(shape),_topology_index(shape)['vertices']) if vv.isSame(v)), str(ref)),
        'point': point,
        'point_norm': _bbox_normalized_point(shape,c,bb),
    }


def match_face(shape,ref):
    records=face_records(shape); faces=_topology_index(shape)['faces']
    # 1. Exact hash or semantic aliases.
    rid=ref.get('id') if isinstance(ref,dict) else ref
    sems=_ref_semantics(ref)
    allowed=_lineage_allowed(shape,'face',ref)
    for i,(r,f) in enumerate(zip(records,faces)):
        if allowed is not None and i not in allowed: continue
        if r['id']==rid or any(x==r.get('semantic') or x in (r.get('semantics') or []) for x in sems): return f
    if not isinstance(ref,dict): raise ValueError(f'Face reference not found: {ref}')
    # 2. Persistent-topology fallback.  Match topology class/orientation first, then
    # choose the closest relative bbox location.  Absolute dimensions are only a weak tie-breaker.
    candidates=[]
    for i,(r,f) in enumerate(zip(records,faces)):
        if allowed is not None and i not in allowed: continue
        if ref.get('geom') and r.get('geom')!=ref.get('geom'): continue
        if ref.get('direction') and r.get('direction')!=ref.get('direction'): continue
        if ref.get('axis_direction') and r.get('axis_direction')!=ref.get('axis_direction'): continue
        if ref.get('radius_mm') is not None and r.get('radius_mm') is not None:
            rr=float(ref['radius_mm']); tol=max(0.05,abs(rr)*0.08)
            if allowed is None and abs(float(r['radius_mm'])-rr)>tol: continue
        if ref.get('edge_count') is not None and r.get('edge_count')!=ref.get('edge_count'):continue
        if ref.get('edge_geom_signature') and r.get('edge_geom_signature')!=ref.get('edge_geom_signature'):continue
        score=0.0
        if ref.get('center_norm') is not None:
            score += float(np.linalg.norm(np.asarray(r.get('center_norm',[0,0,0]),float)-np.asarray(ref['center_norm'],float)))*20.0
        if ref.get('normal') is not None:
            a=np.asarray(r.get('normal',[0,0,0]),float); b=np.asarray(ref['normal'],float)
            if np.linalg.norm(a)>0 and np.linalg.norm(b)>0: score += (1.0-abs(float(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)))))*10.0
        if ref.get('area') not in (None,0): score += abs(math.log(max(float(r.get('area',1e-12)),1e-12)/max(float(ref['area']),1e-12)))
        if allowed is not None and ref.get('radius_mm') not in (None,0) and r.get('radius_mm') not in (None,0):
            score += abs(math.log(max(float(r['radius_mm']),1e-12)/max(float(ref['radius_mm']),1e-12)))
        candidates.append((score,r,f))
    if not candidates: raise ValueError(f'Face reference could not be rebound after topology update: {ref}')
    candidates.sort(key=lambda x:x[0])
    if len(candidates)>1 and abs(candidates[1][0]-candidates[0][0])<1e-8:
        raise ValueError(f'Ambiguous face reference after topology update; refine/reselect. candidates={[x[1].get("id") for x in candidates[:4]]}')
    return candidates[0][2]


def match_edge(shape,ref):
    records=edge_records(shape); edges=_topology_index(shape)['edges']
    rid=ref.get('id') if isinstance(ref,dict) else ref
    allowed=_lineage_allowed(shape,'edge',ref)
    for i,(r,e) in enumerate(zip(records,edges)):
        if allowed is not None and i not in allowed: continue
        if r['id']==rid:return e
    if not isinstance(ref,dict): raise ValueError(f'Edge reference not found: {ref}')
    candidates=[]
    for i,(r,e) in enumerate(zip(records,edges)):
        if allowed is not None and i not in allowed: continue
        if ref.get('geom') and r.get('geom')!=ref.get('geom'): continue
        if ref.get('direction') and r.get('direction')!=ref.get('direction'): continue
        if ref.get('radius_mm') is not None and r.get('radius_mm') is not None:
            rr=float(ref['radius_mm']); tol=max(0.05,abs(rr)*0.08)
            if allowed is None and abs(float(r['radius_mm'])-rr)>tol: continue
        if ref.get('adjacent_face_count') is not None and r.get('adjacent_face_count')!=ref.get('adjacent_face_count'):continue
        if ref.get('adjacent_face_geoms') and r.get('adjacent_face_geoms')!=ref.get('adjacent_face_geoms'):continue
        score=0.0
        if ref.get('center_norm') is not None:
            score += float(np.linalg.norm(np.asarray(r.get('center_norm',[0,0,0]),float)-np.asarray(ref['center_norm'],float)))*20.0
        if ref.get('length') not in (None,0): score += abs(math.log(max(float(r.get('length',1e-12)),1e-12)/max(float(ref['length']),1e-12)))
        if allowed is not None and ref.get('radius_mm') not in (None,0) and r.get('radius_mm') not in (None,0):
            score += abs(math.log(max(float(r['radius_mm']),1e-12)/max(float(ref['radius_mm']),1e-12)))
        candidates.append((score,r,e))
    if not candidates: raise ValueError(f'Edge reference could not be rebound after topology update: {ref}')
    candidates.sort(key=lambda x:x[0])
    if len(candidates)>1 and abs(candidates[1][0]-candidates[0][0])<1e-8:
        raise ValueError(f'Ambiguous edge reference after topology update; refine/reselect. candidates={[x[1].get("id") for x in candidates[:4]]}')
    return candidates[0][2]


def match_edge_ids(shape,ids):
    return [match_edge(shape,x) for x in ids]


def match_vertex_id(shape,ref):
    recs=vertex_records(shape); verts=_topology_index(shape)['vertices']
    rid=ref.get('id') if isinstance(ref,dict) else ref
    for r,v in zip(recs,verts):
        if r['id']==rid:return v
    if not isinstance(ref,dict):
        raise ValueError(f'Vertex reference not found: {ref}')
    # Persistent fallback: relative bbox location survives uniform/parametric size
    # changes substantially better than an absolute point.  Ambiguous ties are
    # rejected rather than silently selecting the wrong topology.
    target_norm=ref.get('point_norm')
    target_abs=ref.get('point')
    candidates=[]
    bb=shape.BoundingBox()
    for r,v in zip(recs,verts):
        c=v.Center(); score=0.0
        if target_norm is not None:
            score += float(np.linalg.norm(np.asarray(_bbox_normalized_point(shape,c,bb),float)-np.asarray(target_norm,float)))*20.0
        if target_abs is not None:
            diag=max(1e-9,math.sqrt(bb.xlen**2+bb.ylen**2+bb.zlen**2))
            score += float(np.linalg.norm(np.asarray([c.x,c.y,c.z],float)-np.asarray(target_abs,float)))/diag
        candidates.append((score,r,v))
    if not candidates:raise ValueError(f'Vertex reference could not be rebound after topology update: {ref}')
    candidates.sort(key=lambda x:x[0])
    if len(candidates)>1 and abs(candidates[1][0]-candidates[0][0])<1e-8:
        raise ValueError(f'Ambiguous vertex reference after topology update; refine/reselect. candidates={[x[1].get("id") for x in candidates[:4]]}')
    return candidates[0][2]


def select_face(shape,spec):
    if shape is None: raise ValueError('No body')
    if isinstance(spec,str): return match_face(shape,spec)
    if not isinstance(spec,dict): raise ValueError('Face selector must be an object')
    # Descriptor refs from persisted feature history take precedence over selector dictionaries.
    if spec.get('id') or spec.get('lineage_owners') or spec.get('semantic') or spec.get('semantics'):
        try:return match_face(shape,spec)
        except ValueError: pass
    kind=str(spec.get('kind','planar')).lower(); records=face_records(shape); faces=_topology_index(shape)['faces']; candidates=[]
    if kind=='planar':
        want=spec.get('normal'); extreme=str(spec.get('extreme','max')).lower(); tol=math.cos(math.radians(float(spec.get('tolerance_deg',5.0))))
        dmap={'+X':np.array([1,0,0]),'-X':np.array([-1,0,0]),'+Y':np.array([0,1,0]),'-Y':np.array([0,-1,0]),'+Z':np.array([0,0,1]),'-Z':np.array([0,0,-1])}
        if want not in dmap: raise ValueError('planar selector normal must be +X|-X|+Y|-Y|+Z|-Z')
        w=dmap[want]
        for r,f in zip(records,faces):
            if r['geom']!='PLANE':continue
            n=np.asarray(r.get('normal',[0,0,0]),float)
            if np.linalg.norm(n)>0 and np.dot(n/np.linalg.norm(n),w)>=tol:candidates.append((r,f))
        if candidates:
            vals=[float(np.dot(np.asarray(r['center']),w)) for r,_ in candidates]; target=(max(vals) if extreme=='max' else min(vals)); eps=max(1e-6,1e-6*max(1,abs(target)))
            candidates=[rf for rf,v in zip(candidates,vals) if abs(v-target)<=eps]
    elif kind=='cylindrical':
        rad=spec.get('radius_mm'); axis=spec.get('axis'); rtol=float(spec.get('radius_tol_mm',0.01)); dmap={'+X':np.array([1,0,0]),'-X':np.array([-1,0,0]),'+Y':np.array([0,1,0]),'-Y':np.array([0,-1,0]),'+Z':np.array([0,0,1]),'-Z':np.array([0,0,-1])}
        for r,f in zip(records,faces):
            if r['geom']!='CYLINDER':continue
            if rad is not None and abs(float(r.get('radius_mm',1e99))-float(rad))>rtol:continue
            if axis in dmap:
                av=np.asarray(r.get('axis',[0,0,0]),float)
                if np.linalg.norm(av)==0 or abs(float(np.dot(av/np.linalg.norm(av),dmap[axis])))<math.cos(math.radians(float(spec.get('tolerance_deg',5)))):continue
            candidates.append((r,f))
    else: raise ValueError('selector kind must be planar or cylindrical')
    near=spec.get('near_mm')
    if len(candidates)>1 and near is not None:
        q=np.asarray(near,float); candidates.sort(key=lambda rf:float(np.linalg.norm(np.asarray(rf[0]['center'])-q)))
        if len(candidates)>=2:
            d0=np.linalg.norm(np.asarray(candidates[0][0]['center'])-q); d1=np.linalg.norm(np.asarray(candidates[1][0]['center'])-q)
            if abs(d0-d1)>1e-8:return candidates[0][1]
    if len(candidates)!=1:
        summary=[{'id':r['id'],'center':r['center'],'normal':r.get('normal'),'radius_mm':r.get('radius_mm'),'axis':r.get('axis')} for r,_ in candidates]
        raise ValueError(f'Face selector matched {len(candidates)} faces; refine with near_mm. candidates={summary}')
    return candidates[0][1]


def face_frame(face):
    c=face.Center(); geom=str(face.geomType())
    if geom=='PLANE':
        n=face.normalAt(c); return {'kind':'plane','origin':[c.x,c.y,c.z],'direction':[n.x,n.y,n.z]}
    if geom=='CYLINDER':
        cyl=face._geomAdaptor().Cylinder(); ax=cyl.Axis(); loc=ax.Location(); d=ax.Direction(); return {'kind':'axis','origin':[loc.X(),loc.Y(),loc.Z()],'direction':[d.X(),d.Y(),d.Z()]}
    raise ValueError(f'Unsupported iMate face geometry: {geom}')

# --- General profile/loop selection helpers ---------------------------------
# These helpers are intentionally feature-agnostic.  They turn an exact edge
# selection into a closed boundary loop and preserve an optional 3D pick point
# as a section anchor.  Loft consumes this today, but the same resolver can be
# reused by any future profile-driven feature.

def _shape_is_same(a, b):
    try:
        return bool(a.isSame(b))
    except Exception:
        return False


def _wire_contains_edge(wire, edge):
    return any(_shape_is_same(candidate, edge) for candidate in wire.Edges())


def _nearest_edge_fraction(edge, point, samples=96):
    """Return a stable normalized [0,1] location on *edge* nearest *point*.

    Browser edge picking is performed on a tessellated polyline, so the hit
    point can be slightly off the analytic B-Rep curve.  Sampling + local
    golden-section refinement avoids requiring the point to lie exactly on the
    OCCT curve while remaining geometry-type agnostic (circle, ellipse, spline,
    line, ...).
    """
    if point is None:
        return None
    try:
        q=np.asarray(point,float)
        if q.shape!=(3,):
            return None
    except Exception:
        return None

    count=max(16,int(samples))
    vals=np.linspace(0.0,1.0,count+1)
    best_t=0.0; best_d=float('inf'); best_i=0
    for i,t in enumerate(vals):
        try:
            v=edge.positionAt(float(t)); a=np.asarray([v.x,v.y,v.z],float)
            d=float(np.dot(a-q,a-q))
        except Exception:
            continue
        if d<best_d:
            best_d=d; best_t=float(t); best_i=i
    lo=float(vals[max(0,best_i-1)]); hi=float(vals[min(count,best_i+1)])
    phi=(1.0+5.0**0.5)/2.0
    for _ in range(24):
        if hi-lo<1e-9:break
        c=hi-(hi-lo)/phi; d=lo+(hi-lo)/phi
        try:
            vc=edge.positionAt(c); vd=edge.positionAt(d)
            ac=np.asarray([vc.x,vc.y,vc.z],float); ad=np.asarray([vd.x,vd.y,vd.z],float)
            dc=float(np.dot(ac-q,ac-q)); dd=float(np.dot(ad-q,ad-q))
        except Exception:
            break
        if dc<=dd:hi=d
        else:lo=c
    return max(0.0,min(1.0,(lo+hi)*0.5))


def reanchor_closed_wire(wire, source_edge=None, anchor_point=None):
    """Return *wire* with its traversal origin moved toward the selected anchor.

    No geometry is changed.  Existing oriented edges are only reordered, and
    the selected edge is split at the anchor when that can be done safely.  If
    OCCT cannot rebuild an equivalent closed wire, the original wire is returned.
    This makes the operation safe-by-fallback rather than replacing the modeling
    backend.
    """
    if wire is None or not wire.IsClosed() or source_edge is None:
        return wire
    edges=list(wire.Edges())
    if not edges:return wire
    idx=next((i for i,e in enumerate(edges) if _shape_is_same(e,source_edge)),None)
    if idx is None:return wire
    oriented=edges[idx]
    ordered=edges[idx:]+edges[:idx]
    t=_nearest_edge_fraction(oriented,anchor_point)
    if t is None or t<=1e-5 or t>=1.0-1e-5:
        try:
            candidate=__import__('cadquery').Wire.assembleEdges(ordered)
            return candidate if candidate.IsClosed() else wire
        except Exception:
            return wire
    try:
        u0=oriented.paramAt(0.0); u1=oriented.paramAt(1.0); ua=oriented.paramAt(float(t))
        # Move the seam to the click point while preserving the original edge
        # orientation around the complete loop.
        tail=oriented.trim(ua,u1)
        head=oriented.trim(u0,ua)
        rebuilt=[tail]+ordered[1:]+[head]
        candidate=__import__('cadquery').Wire.assembleEdges(rebuilt)
        if not candidate.IsClosed():return wire
        old_len=float(wire.Length()); new_len=float(candidate.Length())
        if abs(new_len-old_len)>max(1e-7,old_len*1e-7):return wire
        return candidate
    except Exception:
        return wire


def resolve_edge_loop(shape, edge_ref, anchor_point=None):
    """Resolve one selected edge to an exact closed profile loop.

    Resolution rules are deliberately conservative:
      * a closed periodic edge is itself a valid one-edge loop;
      * otherwise, use the unique closed face loop containing the edge;
      * ambiguous topology is rejected rather than guessed.

    Returns ``(wire, descriptor)``.  The descriptor is JSON-safe and suitable
    for feature history / planner context.
    """
    if shape is None:raise ValueError('No body')
    edge=match_edge(shape,edge_ref)
    persistent=make_edge_ref(shape,edge_ref)
    anchor=None
    if anchor_point is not None:
        try:
            vals=[float(x) for x in anchor_point]
            if len(vals)==3 and all(math.isfinite(x) for x in vals):anchor=vals
        except Exception:pass

    if edge.IsClosed():
        import cadquery as cq
        wire=cq.Wire.assembleEdges([edge])
        wire=reanchor_closed_wire(wire,edge,anchor)
        return wire,{'source':'closed_edge','edge_ref':persistent,'anchor_point':anchor}

    candidates=[]
    for face_rec,face in zip(face_records(shape),_topology_index(shape)['faces']):
        loops=[('outer',face.outerWire())]+[(i+1,w) for i,w in enumerate(face.innerWires())]
        for loop_id,wire in loops:
            if not wire.IsClosed() or not _wire_contains_edge(wire,edge):continue
            candidates.append((face_rec,face,loop_id,wire))
    if not candidates:
        raise ValueError('Selected edge is not part of a closed profile loop')
    # Same geometric loop can occasionally be exposed through duplicate wrappers;
    # remove exact duplicates before deciding whether topology is ambiguous.
    unique=[]
    for row in candidates:
        wire=row[3]
        if any(_shape_is_same(wire,other[3]) for other in unique):continue
        unique.append(row)
    candidates=unique
    if len(candidates)!=1:
        summary=[{'face_ref':r[0].get('id'),'face_geom':r[0].get('geom'),'loop':r[2]} for r in candidates]
        raise ValueError(f'Selected edge belongs to multiple closed profile loops; select a closed edge or a face/loop explicitly. candidates={summary}')
    face_rec,face,loop_id,wire=candidates[0]
    selected_in_wire=next((e for e in wire.Edges() if _shape_is_same(e,edge)),edge)
    wire=reanchor_closed_wire(wire,selected_in_wire,anchor)
    return wire,{
        'source':'face_edge_loop','edge_ref':persistent,'face_ref':make_face_ref(shape,face_rec['id']),
        'loop':loop_id,'anchor_point':anchor,
    }

