"""Opt-in global manufacturing directory, using the same search plan worldwide.

No public Overpass endpoint, payments, or persistent location storage.  API keys
are supplied only as backend environment variables. Outbound requests consume a
persistent, transactionally reserved local budget before leaving the server.

Official APIs: TomTom Places Discover v3; Geoapify Places v2.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
import re
import sqlite3
import threading
import time
from typing import Any

import httpx

_LOG = logging.getLogger(__name__)
_TOMTOM = 'https://api.tomtom.com/maps/orbis/places/discover'
_GEOAPIFY = 'https://api.geoapify.com/v2/places'
_GLOBAL_TERMS = ('CNC', '3D printing', 'rapid prototyping', 'machine shop')
_GLOBAL_POI_TYPES = ('manufacturing_facility','workshop','mechanical_engineering','industrial_facility','industrial_area','construction_business')
_CATEGORY_HINTS = (
    'manufacturing facility', 'manufacturing company', 'mechanical engineering',
    'industrial building', 'factory', 'machine shop', 'machining', 'cnc',
    '3d printing', '3-d printing', 'rapid prototyping', 'prototype',
    'metal fabrication', 'metal working', 'tool and die',
)
_GEO_CATS = 'production.factory,activity.hackerspace,service.metal_construction,service.blacksmith'
_MAX_TOMTOM_MONTH = 2000  # Conservative application-side cap for TomTom calls.
_MAX_GEOAPIFY_DAY = 2400   # Optional fallback only.
_TTL = 15 * 60
_CACHE: dict[tuple[int,int,int], tuple[float,dict[str,Any]]] = {}
_LOCATION_CACHE: dict[str, tuple[float,list[dict[str,Any]]]] = {}
_LOCK = threading.Lock()


class SearchProblem(RuntimeError):
    def __init__(self, public: str, code: str):
        super().__init__(public)
        self.code = code


def distance_km(lat: float, lon: float, a: float, b: float) -> float:
    r1,r2 = math.radians(lat),math.radians(a)
    dlat,dlon = r2-r1,math.radians(b-lon)
    h = math.sin(dlat/2)**2 + math.cos(r1)*math.cos(r2)*math.sin(dlon/2)**2
    return round(6371*2*math.asin(min(1,math.sqrt(h))),2)


def clean(x: Any, size: int=180) -> str:
    return re.sub(r'[\x00-\x1f\x7f]', ' ', str(x or ''))[:size].strip()


def osm_link(lat: float, lon: float) -> str:
    return f'https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=16/{lat:.6f}/{lon:.6f}'


def quota_take(provider: str, credits: int=1) -> bool:
    """Durable reservation across app processes. Fail closed if storage unavailable.

    Counts failed attempts too. This protects only outbound requests made by CADia,
    not use of the same API account/key by other applications.
    """
    if provider not in ('tomtom','geoapify') or not 1 <= credits <= 8:
        raise ValueError('invalid quota reservation')
    now = dt.datetime.now(dt.timezone.utc)
    period = now.strftime('%Y-%m') if provider=='tomtom' else now.strftime('%Y-%m-%d')
    limit = _MAX_TOMTOM_MONTH if provider=='tomtom' else _MAX_GEOAPIFY_DAY
    path = os.environ.get('CADIA_MARKET_QUOTA_DB','/data/cadia-market-quotas.sqlite3')
    con = sqlite3.connect(path, timeout=10)
    try:
        con.execute('PRAGMA busy_timeout=10000')
        con.execute('BEGIN IMMEDIATE')
        con.execute('CREATE TABLE IF NOT EXISTS calls (provider TEXT NOT NULL, period TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(provider, period))')
        count_row = con.execute('SELECT count FROM calls WHERE provider=? AND period=?',(provider,period)).fetchone()
        count = count_row[0] if count_row else 0
        if count+credits>limit:
            con.rollback()
            return False
        con.execute('INSERT INTO calls(provider,period,count) VALUES (?,?,?) ON CONFLICT(provider,period) DO UPDATE SET count=excluded.count', (provider,period,count+credits))
        con.commit()
        return True
    finally:
        con.close()


def normalize_tomtom(response: Any, lat: float, lon: float, radius_m: int, label: str) -> list[dict[str,Any]]:
    if not isinstance(response,dict):
        raise SearchProblem('Place directory returned an unexpected response.','tomtom_schema')
    # TomTom may return an empty JSON object for a valid HTTP 200 search with no
    # selected result attributes. Treat this as zero matches, not as an API failure.
    results=response.get('results',[])
    if not isinstance(results,list):
        raise SearchProblem('Place directory returned an unexpected response.','tomtom_schema')
    output=[]
    for r in results:
        if not isinstance(r,dict) or r.get('type')!='poi': continue
        p=r.get('position') or {}
        coords=p.get('coordinates') if isinstance(p,dict) else None
        if not isinstance(coords,list) or len(coords)<2:continue
        try:
            b,a=float(coords[0]),float(coords[1])
        except (ValueError,TypeError):continue
        if not all(math.isfinite(v) for v in (a,b)) or abs(a)>90 or abs(b)>180:continue
        d=distance_km(lat,lon,a,b)
        if d*1000>radius_m+30:continue
        name=clean(r.get('title'),120)
        if not name:continue
        subs=r.get('subtitles') or []
        address=' · '.join(clean(x,120) for x in subs[:2] if isinstance(x,str)) if isinstance(subs,list) else ''
        output.append({'id':'tomtom:'+clean(r.get('id') or f'{a:.5f}:{b:.5f}:{name}',130),
                       'name':name,'lat':a,'lon':b,'distance_km':d,'address':address,
                       'process':label+' (capabilities unverified)', 'source':'TomTom Places', 'url':osm_link(a,b)})
    return output


def normalize_geoapify(response: Any, lat: float, lon: float, radius_m: int) -> list[dict[str,Any]]:
    if not isinstance(response,dict) or not isinstance(response.get('features'),list):
        raise SearchProblem('Place directory returned an unexpected response.','geoapify_schema')
    output=[]
    for f in response['features']:
        if not isinstance(f,dict):continue
        p=f.get('properties') or {}; g=f.get('geometry') or {}
        if not isinstance(p,dict) or not isinstance(g,dict):continue
        c=g.get('coordinates') or []
        if not isinstance(c,list) or len(c)<2:continue
        try:b,a=float(c[0]),float(c[1])
        except (ValueError,TypeError):continue
        if not all(math.isfinite(v) for v in (a,b)) or abs(a)>90 or abs(b)>180:continue
        d=distance_km(lat,lon,a,b)
        if d*1000>radius_m+30:continue
        name=clean(p.get('name') or p.get('address_line1'),120)
        if not name:continue
        cats=p.get('categories') or []
        cat_text=' '.join(cats) if isinstance(cats,list) else ''
        if 'hackerspace' in cat_text:label='Makerspace'
        elif 'metal_construction' in cat_text or 'blacksmith' in cat_text:label='Metal fabrication'
        else:label='Manufacturing'
        output.append({'id':'geoapify:'+clean(p.get('place_id') or f'{a:.5f}:{b:.5f}:{name}',130),
                       'name':name,'lat':a,'lon':b,'distance_km':d,
                       'address':clean(p.get('formatted') or p.get('address_line2'),200),
                       'process':label+' (capabilities unverified)', 'source':'Geoapify / OpenStreetMap',
                       'url':osm_link(a,b)})
    return output


def _places_json(resp: httpx.Response, provider: str) -> dict[str, Any]:
    _LOG.info('manufacturing_search_provider provider=%s status=%d', provider, resp.status_code)
    if resp.status_code in (401, 403):
        raise SearchProblem('The TomTom Places Search API key is not authorized for this request.', 'places_access')
    if resp.status_code == 429:
        raise SearchProblem('The supplier directory is rate-limiting requests.', 'rate_limit')
    if resp.status_code >= 400:
        raise SearchProblem('The supplier directory is temporarily unavailable.', 'upstream_http_'+str(resp.status_code))
    try:
        data = resp.json()
    except ValueError as exc:
        raise SearchProblem('The supplier directory returned invalid JSON.', 'invalid_json') from exc
    if not isinstance(data, dict):
        raise SearchProblem('The supplier directory returned an unexpected response.', 'bad_json')
    return data


def _places_headers(api_key: str) -> dict[str, str]:
    # Do not force one country language. TomTom falls back to the prevalent local
    # language when Accept-Language is omitted.
    return {
        'TomTom-Api-Key': api_key,
        'TomTom-Api-Version': '3',
        'Attributes': 'results',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }


def _places_body(lat: float, lon: float, radius_m: int, *, query: str | None = None, poi_types: tuple[str, ...] = ()) -> dict[str, Any]:
    filters: dict[str, Any] = {
        'types': ['poi'],
        'geometry': {'type': 'circle', 'center': [lon, lat], 'radiusInMeters': int(radius_m)},
    }
    if poi_types:
        filters['poiTypes'] = list(poi_types)
    # Places v3 supports category/geometry discovery without a text query.  Do not
    # send query='' (the service rejects an explicitly empty query with HTTP 400).
    # Omitting the field keeps the primary supplier search language-neutral.
    body: dict[str, Any] = {
        'origin': {'type': 'point', 'coordinates': [lon, lat]},
        'maxResults': 100,
        'filters': filters,
        'preferences': {'geometry': {'type': 'point', 'coordinates': [lon, lat]}},
    }
    if query is not None:
        q = clean(query, 100)
        if q:
            body['query'] = q
    return body


def _request_places(client: httpx.Client, lat: float, lon: float, radius_m: int, api_key: str, *, query: str | None = None, poi_types: tuple[str, ...] = ()) -> dict[str, Any]:
    if not quota_take('tomtom'):
        raise SearchProblem('CADia has reached its free supplier-search budget. No paid request was made.', 'tomtom_budget')
    resp = client.post(_TOMTOM, headers=_places_headers(api_key), json=_places_body(lat, lon, radius_m, query=query, poi_types=poi_types))
    return _places_json(resp, 'tomtom_places')


def normalize_places(response: Any, lat: float, lon: float, radius_m: int, label: str, priority: int) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        raise SearchProblem('The supplier directory returned an unexpected response.', 'places_schema')
    results = response.get('results', [])
    # Places Discover can legitimately return {} for a successful zero-result query.
    if results is None:
        results = []
    if not isinstance(results, list):
        raise SearchProblem('The supplier directory returned an unexpected response.', 'places_schema')
    out=[]
    for r in results:
        if not isinstance(r, dict) or str(r.get('type','')).casefold() != 'poi':
            continue
        pos=r.get('position') or {}
        coords=pos.get('coordinates') if isinstance(pos,dict) else None
        if not isinstance(coords,list) or len(coords)<2:
            continue
        try: b,a=float(coords[0]),float(coords[1])
        except (TypeError,ValueError): continue
        if not all(math.isfinite(v) for v in (a,b)) or abs(a)>90 or abs(b)>180: continue
        d=distance_km(lat,lon,a,b)
        if d*1000>radius_m+30: continue
        name=clean(r.get('title'),120)
        if not name: continue
        address=''
        addr=r.get('address') or {}
        if isinstance(addr,dict):
            parts=[addr.get('street'),addr.get('houseNumber'),addr.get('municipality'),addr.get('country')]
            address=', '.join(clean(x,90) for x in parts if x)
        if not address:
            subs=r.get('subtitles') or []
            if isinstance(subs,list): address=' · '.join(clean(x,120) for x in subs[:3] if isinstance(x,str))
        names=[]
        for t in r.get('poiTypes') or []:
            if isinstance(t,dict) and t.get('name'): names.append(clean(t.get('name'),60))
        process=' / '.join(dict.fromkeys(names[:2])) or label
        out.append({'id':'tomtom-places:'+clean(r.get('id') or f'{a:.5f}:{b:.5f}:{name}',130),
                    'name':name,'lat':a,'lon':b,'distance_km':d,'address':address,
                    'process':process+' (capabilities unverified)','source':'TomTom Places Search API',
                    'url':osm_link(a,b),'_priority':priority})
    return out


def _request_geoapify(client: httpx.Client, lat: float, lon: float, radius_m: int, api_key: str) -> dict[str,Any]:
    resp=client.get(_GEOAPIFY,params={'categories':_GEO_CATS,'filter':f'circle:{lon:.6f},{lat:.6f},{radius_m}','bias':f'proximity:{lon:.6f},{lat:.6f}','limit':20,'apiKey':api_key})
    _LOG.info('manufacturing_search_provider provider=geoapify status=%d',resp.status_code)
    if resp.status_code in (401,403): raise SearchProblem('The location search API key is invalid or lacks access.','bad_key')
    if resp.status_code==429: raise SearchProblem('The place search provider is rate-limiting requests.','rate_limit')
    if resp.status_code >= 400: raise SearchProblem('The place search provider is temporarily unavailable.','upstream_http_'+str(resp.status_code))
    try: result=resp.json()
    except ValueError as e: raise SearchProblem('The place search provider returned invalid JSON.','invalid_json') from e
    if not isinstance(result,dict): raise SearchProblem('The place search provider returned an unexpected response.','bad_json')
    return result


def _dedup(entries: list[dict[str,Any]]) -> list[dict[str,Any]]:
    result=[]; seen=set()
    for e in sorted(entries,key=lambda p:(int(p.get('_priority',1)), p['distance_km'])):
        key=(e['name'].casefold().strip(),round(e['lat'],3),round(e['lon'],3))
        if key in seen: continue
        seen.add(key); e=dict(e); e.pop('_priority',None); result.append(e)
    return result[:50]


def search_locations(query: str) -> list[dict[str,Any]]:
    """Resolve a user-entered city/address with the same TomTom Places key.

    This is intentionally separate from nearby supplier discovery: the user may
    choose any search center without granting browser geolocation permission.
    No query or resolved location is persisted.
    """
    q=clean(query,160)
    if len(q)<2:
        raise SearchProblem('Enter at least two characters for a city or address.','location_query')
    cache_key=q.casefold()
    with _LOCK:
        hit=_LOCATION_CACHE.get(cache_key)
        if hit and time.monotonic()-hit[0] < 60*60:
            return hit[1]
    key=os.getenv('CADIA_TOMTOM_API_KEY','').strip()
    if not key:
        raise SearchProblem('Location search is not configured.','unconfigured')
    try:
        if not quota_take('tomtom'):
            raise SearchProblem('CADia has reached its free location-search budget. No paid request was made.','budget')
        with httpx.Client(timeout=httpx.Timeout(12.0,connect=4.0),follow_redirects=False,
                          headers={'User-Agent':'CADia-Manufacturing-Finder/1.4 (https://cadia.co.kr)'}) as client:
            response=client.post(_TOMTOM,headers={
                'TomTom-Api-Key':key,'TomTom-Api-Version':'3','Attributes':'results(id,type,title,position,address(countryCodeIso2))',
                'Content-Type':'application/json','Accept':'application/json','Accept-Language':'en-US,en;q=0.9',
            },json={'query':q,'maxResults':8,'filters':{'types':['area','address','street','poi']}})
    except httpx.RequestError as exc:
        _LOG.warning('manufacturing_location_search_failure kind=%s',type(exc).__name__)
        raise SearchProblem('Location lookup is temporarily unavailable.','location_network') from exc
    _LOG.info('manufacturing_location_search status=%d',response.status_code)
    if response.status_code in (401,403):
        raise SearchProblem('The location search API key is invalid or lacks access.','bad_key')
    if response.status_code==429:
        raise SearchProblem('The place search provider is rate-limiting requests.','rate_limit')
    if response.status_code>=400:
        raise SearchProblem('Location lookup is temporarily unavailable.','location_http_'+str(response.status_code))
    try:
        data=response.json()
    except ValueError as exc:
        raise SearchProblem('Location lookup returned invalid data.','location_json') from exc
    results=data.get('results',[]) if isinstance(data,dict) else []
    if not isinstance(results,list):
        raise SearchProblem('Location lookup returned an unexpected response.','location_schema')
    output=[]
    for row in results:
        if not isinstance(row,dict):
            continue
        pos=row.get('position') or {}
        coords=pos.get('coordinates') if isinstance(pos,dict) else None
        if not isinstance(coords,list) or len(coords)<2:
            continue
        try:
            lon,lat=float(coords[0]),float(coords[1])
        except (TypeError,ValueError):
            continue
        if not all(math.isfinite(v) for v in (lat,lon)) or abs(lat)>90 or abs(lon)>180:
            continue
        title=clean(row.get('title'),120)
        if not title:
            continue
        address=row.get('address') or {}
        country_code=clean(address.get('countryCodeIso2'),2).upper() if isinstance(address,dict) else ''
        output.append({'id':clean(row.get('id') or f'{lat:.5f}:{lon:.5f}:{title}',150),
                       'label':title,'subtitle':country_code,'country_code':country_code,'lat':lat,'lon':lon,'kind':clean(row.get('type'),30)})
    # De-duplicate near-identical geocoding results while preserving provider rank.
    seen=set(); dedup=[]
    for row in output:
        k=(row['label'].casefold(),round(row['lat'],4),round(row['lon'],4))
        if k in seen: continue
        seen.add(k); dedup.append(row)
        if len(dedup)>=6: break
    with _LOCK:
        if len(_LOCATION_CACHE)>=64: _LOCATION_CACHE.clear()
        _LOCATION_CACHE[cache_key]=(time.monotonic(),dedup)
    return dedup


def find_nearby(lat:float, lon:float, radius_m:int) -> dict[str,Any]:
    key=(round(lat*1000),round(lon*1000),radius_m)
    with _LOCK:
        hit=_CACHE.get(key)
        if hit and time.monotonic()-hit[0]<_TTL:return hit[1]
    tom=os.getenv('CADIA_TOMTOM_API_KEY','').strip(); geo=os.getenv('CADIA_GEOAPIFY_API_KEY','').strip()
    if not tom and not geo:
        raise SearchProblem('Supplier search is not configured. Ask the administrator to set a free-tier API key.','unconfigured')
    errors=[]; shops=[]; used=[]; tested=False
    try:
        with httpx.Client(timeout=httpx.Timeout(14.0,connect=4.0),follow_redirects=False,headers={'User-Agent':'CADia-Supplier-Finder/1.9 (https://cadia.co.kr)'}) as client:
            if tom:
                # Primary path: category + radius only, with NO text query.  This is
                # identical worldwide and does not depend on English/Korean/Japanese/etc.
                try:
                    data=_request_places(client,lat,lon,radius_m,tom,poi_types=_GLOBAL_POI_TYPES)
                    tested=True; used.append('TomTom Places Search API')
                    shops.extend(normalize_places(data,lat,lon,radius_m,'Manufacturing / engineering',0))
                except (SearchProblem,httpx.RequestError) as e:
                    code=e.code if isinstance(e,SearchProblem) else type(e).__name__; errors.append(code); _LOG.warning('manufacturing_search_failure provider=places_categories code=%s',code)
                # Secondary recall pass only.  The main search above is language-neutral;
                # these globally shared technical terms are never selected by country/region.
                if len(_dedup(shops)) < 24:
                    for q in _GLOBAL_TERMS:
                        try:
                            data=_request_places(client,lat,lon,radius_m,tom,query=q)
                            tested=True
                            if 'TomTom Places Search API' not in used: used.append('TomTom Places Search API')
                            shops.extend(normalize_places(data,lat,lon,radius_m,q,0))
                            if len(_dedup(shops)) >= 36: break
                        except (SearchProblem,httpx.RequestError) as e:
                            code=e.code if isinstance(e,SearchProblem) else type(e).__name__; errors.append(code); _LOG.warning('manufacturing_search_failure provider=places_text code=%s search=%s',code,q)
                            if code == 'places_access': break
            if geo and not shops:
                try:
                    if not quota_take('geoapify',1): errors.append('geoapify_budget')
                    else:
                        data=_request_geoapify(client,lat,lon,radius_m,geo); tested=True; used.append('Geoapify / OpenStreetMap')
                        for e in normalize_geoapify(data,lat,lon,radius_m): e['_priority']=2; shops.append(e)
                except (SearchProblem,httpx.RequestError) as e:
                    code=e.code if isinstance(e,SearchProblem) else type(e).__name__; errors.append(code); _LOG.warning('manufacturing_search_failure provider=geoapify code=%s',code)
    except (OSError,sqlite3.Error) as e:
        _LOG.error('manufacturing_quota_storage_failure kind=%s',type(e).__name__); raise SearchProblem('Free search limit protection is unavailable. Search has been paused.','quota_db') from e
    if not tested:
        if 'places_access' in errors: raise SearchProblem('TomTom Places Search API is not authorized for this key.','places_access')
        if any(x.endswith('_budget') for x in errors): raise SearchProblem('CADia has reached its free supplier-search budget. No paid request was made.','budget')
        if 'bad_key' in errors: raise SearchProblem('The location search API key is invalid.','bad_key')
        raise SearchProblem('The supplier directory is unavailable. Try again later.','unavailable')
    shops=_dedup(shops)
    response={'providers':shops,'source':' + '.join(dict.fromkeys(used)),'coverage':'Directory matches only. Manufacturing/CNC/3D-printing capability and order acceptance have not been verified.','partial_results':bool(errors),'attribution':'Powered by Geoapify · © OpenStreetMap contributors' if 'Geoapify / OpenStreetMap' in used else ''}
    with _LOCK:
        if len(_CACHE)>=128:_CACHE.clear()
        _CACHE[key]=(time.monotonic(),response)
    return response

