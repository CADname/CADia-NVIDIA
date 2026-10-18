import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  CheckCircle2, ExternalLink, Factory, LocateFixed, MapPin, Minus, Plus,
  Printer, RefreshCw, Search, ShieldCheck, TriangleAlert,
} from 'lucide-react'
import { api } from '../api'
import '../marketplace.css'
import { materialOptions, quoteFields } from './marketplacePresentation'

type Shop = { id: string; name: string; lat: number; lon: number; distance_km: number; address: string; process: string; source: string; url: string }
type Point = { lat: number; lon: number }
type LocationMatch = { id: string; label: string; subtitle: string; country_code?: string; lat: number; lon: number; kind: string }
type QuoteEnvelope = { provider: string; stage: string; payment_enabled: boolean; result: unknown }
type Props = { projectId: string; modelReady?: boolean; mode: 'suppliers' | 'quote' }

const TILE = 256
const MAXLAT = 85.05112878
const clampLat = (n:number) => Math.max(-MAXLAT, Math.min(MAXLAT, n))
const project = (lat:number,lon:number,zoom:number) => {
  const scale = TILE * 2**zoom
  const safe = clampLat(lat)*Math.PI/180
  return { x: (lon+180)/360*scale, y: (1-Math.log(Math.tan(safe)+1/Math.cos(safe))/Math.PI)/2*scale }
}
function englishCountryName(code?:string){
  const value=(code||'').trim().toUpperCase()
  if(!value)return ''
  try{
    const DisplayNames=(Intl as typeof Intl & {DisplayNames?: new(locales:string[],options:{type:'region'})=>{of:(code:string)=>string|undefined}}).DisplayNames
    return DisplayNames ? (new DisplayNames(['en'],{type:'region'}).of(value)||value) : value
  }catch{return value}
}

const unproject = (x:number,y:number,zoom:number):Point => {
  const scale = TILE * 2**zoom
  const lon = ((x/scale*360-180+540)%360)-180
  const merc = Math.PI - 2*Math.PI*y/scale
  const lat = 180/Math.PI*Math.atan(Math.sinh(merc))
  return {lat:clampLat(lat),lon}
}

function ShopsMap({center,shops,centerLabel,onUseMyLocation,locating}:{center:Point,shops:Shop[],centerLabel:string,onUseMyLocation:()=>void,locating:boolean}) {
  const host = useRef<HTMLDivElement>(null)
  const drag = useRef<{pointerId:number,startX:number,startY:number,startWorld:{x:number,y:number}}|null>(null)
  const [size,setSize] = useState({width:850,height:350})
  const [zoom,setZoom] = useState(12)
  const [focus,setFocus] = useState<Point>(center)
  const [dragging,setDragging] = useState(false)
  const {width,height}=size
  useEffect(()=>{setFocus(center);setZoom(12)},[center.lat,center.lon])
  useEffect(()=>{
    const el=host.current
    if(!el)return
    const observer=new ResizeObserver(([entry])=>{
      if(!entry)return
      const width=Math.max(260,entry.contentRect.width), height=Math.max(220,entry.contentRect.height)
      setSize(prev=>prev.width===width&&prev.height===height?prev:{width,height})
    })
    observer.observe(el)
    return ()=>observer.disconnect()
  },[])
  const c=project(focus.lat,focus.lon,zoom)
  const left=Math.floor((c.x-width/2)/TILE), right=Math.ceil((c.x+width/2)/TILE)
  const top=Math.floor((c.y-height/2)/TILE), bottom=Math.ceil((c.y+height/2)/TILE)
  const tiles:Array<{key:string,url:string,x:number,y:number}>=[]
  const n=2**zoom
  for(let y=Math.max(0,top);y<=Math.min(n-1,bottom);y++) for(let x=left;x<=right;x++) {
    if(tiles.length>=50) break
    tiles.push({key:`${zoom}-${x}-${y}`,url:`https://tile.openstreetmap.org/${zoom}/${((x%n)+n)%n}/${y}.png`,x:width/2+x*TILE-c.x,y:height/2+y*TILE-c.y})
  }
  const pins=shops.map(p=>{let dx=project(p.lat,p.lon,zoom).x-c.x;const size=TILE*2**zoom;if(dx>size/2)dx-=size;if(dx< -size/2)dx+=size
    return {...p,x:width/2+dx,y:height/2+project(p.lat,p.lon,zoom).y-c.y} })
  const own=project(center.lat,center.lon,zoom)
  const me={x:width/2+own.x-c.x,y:height/2+own.y-c.y}
  return <div className="market-map" ref={host} aria-label="Map of manufacturing suppliers near the selected location"
    style={{touchAction:'none',cursor:dragging?'grabbing':'grab'}}
    onPointerDown={event=>{
      if(event.button!==0)return
      const target=event.target as HTMLElement
      if(target.closest('button,a'))return
      drag.current={pointerId:event.pointerId,startX:event.clientX,startY:event.clientY,startWorld:project(focus.lat,focus.lon,zoom)}
      event.currentTarget.setPointerCapture(event.pointerId)
      setDragging(true)
      event.preventDefault()
    }}
    onPointerMove={event=>{
      const state=drag.current
      if(!state || state.pointerId!==event.pointerId)return
      const dx=event.clientX-state.startX
      const dy=event.clientY-state.startY
      setFocus(unproject(state.startWorld.x-dx,state.startWorld.y-dy,zoom))
      event.preventDefault()
    }}
    onPointerUp={event=>{
      if(drag.current?.pointerId!==event.pointerId)return
      drag.current=null
      setDragging(false)
      if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId)
    }}
    onPointerCancel={event=>{
      if(drag.current?.pointerId!==event.pointerId)return
      drag.current=null
      setDragging(false)
    }}
    onLostPointerCapture={()=>{drag.current=null;setDragging(false)}}>
    {tiles.map(t=><img key={t.key} src={t.url} alt="" draggable={false} referrerPolicy="strict-origin-when-cross-origin" style={{position:'absolute',left:t.x,top:t.y,width:TILE,height:TILE}}/>)}
    {pins.filter(p=>p.x>=-30&&p.x<=width+30&&p.y>=-30&&p.y<=height+30).map(p=><button type="button" key={p.id} className="market-pin" style={{left:p.x,top:p.y}} title={`${p.name} · ${p.distance_km} km`} onClick={()=>setFocus({lat:p.lat,lon:p.lon})}><Printer size={14}/></button>)}
    <div className="market-you" title={`Search center: ${centerLabel}`} style={{left:me.x,top:me.y}}/>
    <div className="market-map-controls" aria-label="Map controls"><button type="button" onClick={()=>setZoom(v=>Math.min(16,v+1))} disabled={zoom>=16} aria-label="Zoom in" title="Zoom in"><Plus size={16}/></button><button type="button" onClick={()=>setZoom(v=>Math.max(6,v-1))} disabled={zoom<=6} aria-label="Zoom out" title="Zoom out"><Minus size={16}/></button><button type="button" className="market-gps-button" onClick={onUseMyLocation} disabled={locating} aria-label="Use my current location" title="Use my current location and search nearby"><LocateFixed size={16}/></button></div>
    <span className="market-map-credit">© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a></span>
  </div>
}

function QuoteSummary({value, type='quote'}:{value:unknown;type?:'quote'|'shipping'}) {
  const fields=quoteFields(value)
  return fields.length ? <dl className="market-summary">{fields.map(item=><div key={item.label}><dt>{item.label}</dt><dd>{item.value}</dd></div>)}</dl>
    : <p className="market-help">{type==='shipping' ? 'Shipping response received. The provider did not supply recognized rate fields.' : 'Provider responded, but did not supply recognized price fields. No amount has been assumed.'}</p>
}
function findQuoteId(input:unknown):string|null {
  if(Array.isArray(input)){for(const item of input){const found=findQuoteId(item);if(found)return found}return null}
  if(!input || typeof input!=='object')return null
  const record=input as Record<string,unknown>
  for(const key of ['quote_id','quoteId','quoteID'])if(typeof record[key]==='string' || typeof record[key]==='number')return String(record[key])
  for(const key of ['quote','data','result','content','text']){
    const nested=record[key]
    if(typeof nested==='string'){try{const parsed=JSON.parse(nested);const value=findQuoteId(parsed);if(value)return value}catch{/* text is not JSON */}}
    else if(nested && typeof nested==='object'){const value=findQuoteId(nested);if(value)return value}
  }
  return null
}

export default function ManufacturingMarketplace({projectId,modelReady,mode}:Props){
  const [point,setPoint] = useState<Point|null>(null)
  const [centerLabel,setCenterLabel] = useState('Selected location')
  const [shops,setShops] = useState<Shop[]>([])
  const [locationBusy,setLocationBusy] = useState(false)
  const [locationError,setLocationError] = useState('')
  const [radius,setRadius] = useState(12000)
  const [searchCompleted,setSearchCompleted] = useState(false)
  const [lastSearchRadius,setLastSearchRadius] = useState(12000)
  const [placeQuery,setPlaceQuery] = useState('')
  const [placeBusy,setPlaceBusy] = useState(false)
  const [placeResults,setPlaceResults] = useState<LocationMatch[]>([])
  const [placeError,setPlaceError] = useState('')

  const [material,setMaterial] = useState('PLA')
  const [color,setColor] = useState('black')
  const [filamentId,setFilamentId] = useState('')
  const [materialsResponse,setMaterialsResponse] = useState<unknown|null>(null)
  const [materialSearch,setMaterialSearch] = useState('')
  const [quantity,setQuantity] = useState(1)
  const [uploadConsent,setUploadConsent] = useState(false)
  const [realQuote,setRealQuote] = useState<QuoteEnvelope|null>(null)
  const [qtyResult,setQtyResult] = useState<QuoteEnvelope|null>(null)
  const [shippingResult,setShippingResult] = useState<QuoteEnvelope|null>(null)
  const [quoteBusy,setQuoteBusy] = useState('')
  const [quoteError,setQuoteError] = useState('')
  const [quoteId,setQuoteId] = useState('')
  const [shipConsent,setShipConsent] = useState(false)
  const [address,setAddress] = useState({name:'',email:'',line1:'',city:'',state:'',postal_code:'',country:''})
  const [shippingService,setShippingService] = useState('')
  const choices=useMemo(()=>materialOptions(materialsResponse),[materialsResponse])
  const filteredChoices=useMemo(()=>choices.filter(x=>`${x.material} ${x.color}`.toLowerCase().includes(materialSearch.trim().toLowerCase())).slice(0,250),[choices,materialSearch])
  const selectedMaterial=choices.find(x=>x.id===filamentId)

  const findNearby = useCallback(async (geo:Point,r:number,label:string)=>{
    setSearchCompleted(false);setShops([]);setLocationError('')
    const response=await api<{providers:Shop[],source:string}>(`/api/manufacturing/market/nearby`,{method:'POST',body:JSON.stringify({...geo,radius_m:r,consent:true})})
    setPoint(geo);setCenterLabel(label);setShops(response.providers);setSearchCompleted(true);setLastSearchRadius(r)
  },[])

  async function locate(){
    setLocationError('');setPlaceError('');setLocationBusy(true);setPlaceResults([])
    if(!navigator.geolocation){setLocationBusy(false);setLocationError('Your browser does not support location sharing. Search for a city or address instead.');return}
    navigator.geolocation.getCurrentPosition(async position=>{
      const geo={lat:position.coords.latitude,lon:position.coords.longitude}
      try{await findNearby(geo,radius,'Current location')}catch(err){setLocationError(err instanceof Error?err.message:String(err))}
      finally{setLocationBusy(false)}
    },err=>{setLocationBusy(false);setLocationError(err.code===1?'Location permission was denied. Search for a city or address instead.':'Unable to determine your location. Search for a city or address instead.')},
    {enableHighAccuracy:false,timeout:14000,maximumAge:60000})
  }

  async function searchPlace(event?:FormEvent){
    event?.preventDefault()
    const q=placeQuery.trim()
    if(q.length<2){setPlaceError('Enter a city, district, or address.');return}
    setPlaceBusy(true);setPlaceError('');setPlaceResults([])
    try{
      const response=await api<{results:LocationMatch[]}>(`/api/manufacturing/market/locations?q=${encodeURIComponent(q)}`)
      setPlaceResults(response.results)
      if(!response.results.length)setPlaceError('No matching location found. Try a city plus country, such as “Busan, South Korea”.')
    }catch(err){setPlaceError(err instanceof Error?err.message:String(err))}finally{setPlaceBusy(false)}
  }

  async function choosePlace(place:LocationMatch){
    setPlaceResults([]);setLocationBusy(true);setLocationError('')
    const country=englishCountryName(place.country_code);const label=country ? `${place.label} · ${country}` : place.label
    try{await findNearby({lat:place.lat,lon:place.lon},radius,label)}catch(err){setLocationError(err instanceof Error?err.message:String(err))}finally{setLocationBusy(false)}
  }

  async function changeRadius(value:number){
    setRadius(value)
    if(point){setLocationBusy(true);setLocationError('');try{await findNearby(point,value,centerLabel)}catch(err){setLocationError(err instanceof Error?err.message:String(err))}finally{setLocationBusy(false)}}
  }

  async function runQuote(){
    setQuoteError('');setQuoteBusy('quote');setRealQuote(null);setQtyResult(null);setShippingResult(null);setQuoteId('')
    try{
      const response=await api<QuoteEnvelope>('/api/manufacturing/market/slant3d/quote',{method:'POST',body:JSON.stringify({project_id:projectId,material,color,filament_id:filamentId,quantity,consent_to_external_transfer:uploadConsent})})
      setRealQuote(response);setQuoteId(findQuoteId(response.result)||'')
    }catch(err){setQuoteError(err instanceof Error?err.message:String(err))}finally{setQuoteBusy('')}
  }
  async function updateQuantity(){
    setQuoteError('');setQuoteBusy('quantity');try{const q=await api<QuoteEnvelope>(`/api/manufacturing/market/slant3d/quantity?quote_id=${encodeURIComponent(quoteId)}&quantity=${quantity}`,{method:'POST'});setQtyResult(q)}catch(err){setQuoteError(err instanceof Error?err.message:String(err))}finally{setQuoteBusy('')}
  }
  async function updateShipping(){
    setQuoteError('');setQuoteBusy('shipping');try{const q=await api<QuoteEnvelope>('/api/manufacturing/market/slant3d/shipping',{method:'POST',body:JSON.stringify({...address,quote_id:quoteId,shipping_service:shippingService || undefined,consent_to_external_transfer:shipConsent})});setShippingResult(q)}catch(err){setQuoteError(err instanceof Error?err.message:String(err))}finally{setQuoteBusy('')}
  }
  async function loadMaterials(){
    setQuoteBusy('materials');setQuoteError('')
    try{
      const v=await api<{result:unknown}>('/api/manufacturing/market/slant3d/materials');setMaterialsResponse(v.result)
      const list=materialOptions(v.result)
      if(list.length){const match=list.find(x=>x.material.toLowerCase()===material.toLowerCase()&&x.color.toLowerCase()===color.toLowerCase())||list.find(x=>x.material.toLowerCase()===material.toLowerCase())||list[0];setFilamentId(match.id);setMaterial(match.material);setColor(match.color)}
    }catch(err){setQuoteError(err instanceof Error?err.message:String(err))}finally{setQuoteBusy('')}
  }

  if(mode==='suppliers') return <section className="market-wrap market-suppliers supplier-finder-content">
    <div className="market-search-card supplier-search-card">
      <div className="market-search-actions">
        <button className="studio-btn accent" disabled={locationBusy} onClick={()=>void locate()}><LocateFixed size={16}/>{locationBusy?'Locating…':'My location'}</button>
        <span className="market-or">or</span>
        <form className="market-place-form" onSubmit={e=>void searchPlace(e)}>
          <Search size={16}/><input value={placeQuery} onChange={e=>setPlaceQuery(e.target.value)} placeholder="Search city or address — Tokyo, Busan, Berlin…" aria-label="Search city or address"/>
          <button className="studio-btn ghost" disabled={placeBusy}>{placeBusy?'Searching…':'Search'}</button>
        </form>
      </div>
      {placeError && <p className="market-inline-error"><TriangleAlert size={15}/>{placeError}</p>}
      {placeResults.length>0 && <div className="market-place-results">{placeResults.map(place=><button key={place.id} onClick={()=>void choosePlace(place)}><MapPin size={15}/><span><strong>{place.label}</strong>{englishCountryName(place.country_code)&&<small>{englishCountryName(place.country_code)}</small>}</span></button>)}</div>}
      <div className="market-search-meta">
        <div><span>Search center</span><strong>{point?centerLabel:'Not selected'}</strong></div>
        <div className="market-radius" aria-label="Search radius"><span>Radius</span>{[5000,12000,25000].map(v=><button key={v} className={radius===v?'active':''} disabled={locationBusy} onClick={()=>void changeRadius(v)}>{v/1000} km</button>)}</div>
      </div>
    </div>

    {locationError && <p className="market-alert"><TriangleAlert size={16}/>{locationError}</p>}
    {!point && <div className="market-start"><LocateFixed size={28}/><strong>Choose a search location</strong><span>GPS is optional. Search any city or address to inspect suppliers there.</span></div>}
    {point && <div className="market-locator-layout">
      <div className="market-map-column"><ShopsMap key={`${point.lat}:${point.lon}`} center={point} shops={shops} centerLabel={centerLabel} onUseMyLocation={()=>void locate()} locating={locationBusy}/></div>
      <aside className="market-result-column"><div className="market-result-head"><div><strong>Nearby results</strong><span>{locationBusy?'Searching…':searchCompleted?`${shops.length} found within ${lastSearchRadius/1000} km`:'Ready'}</span></div></div>
        <div className="market-results">{locationBusy?<p className="market-empty compact">Searching nearby manufacturers…</p>:searchCompleted&&shops.length===0?<div className="market-empty"><strong>No matching directory listings within {lastSearchRadius/1000} km.</strong><span>This does not prove there are no manufacturers nearby. Try another city, a larger radius, or contact local suppliers directly.</span>{lastSearchRadius<25000&&<button className="studio-btn ghost" onClick={()=>void changeRadius(25000)}>Search 25 km</button>}</div>:shops.map(shop=><article className="market-shop" key={shop.id}><span className="market-shop-icon"><Factory size={16}/></span><div><strong>{shop.name}</strong><small>{shop.process} · {shop.distance_km.toLocaleString()} km</small>{shop.address&&<small>{shop.address}</small>}</div><a href={shop.url} target="_blank" rel="noopener noreferrer" aria-label={`Open ${shop.name} on map`}><ExternalLink size={15}/></a></article>)}</div>
      </aside>
    </div>}
  </section>

  return <section className="market-wrap market-quote-mode">
    <header className="market-page-head"><div className="market-page-icon"><Printer size={21}/></div><div><span className="market-kicker">INSTANT QUOTE</span><h3>Get a real 3D-printing quote</h3><p>Send the current STL to Slant 3D only after consent. CADia stops before checkout, payment, or production.</p></div></header>

    <div className="market-quote-setup">
      <div className="market-quote-status"><div><span>Model</span><strong>{modelReady?'Ready to quote':'No printable model'}</strong></div><div><span>Provider</span><strong>Slant 3D</strong></div><div><span>Checkout</span><strong>Disabled</strong></div></div>
      <div className="market-catalog-row"><div><strong>Material & color</strong><span>Use provider catalog when available, or enter values manually.</span></div><button className="studio-btn ghost" disabled={!!quoteBusy} onClick={()=>void loadMaterials()}>{quoteBusy==='materials'?'Loading…':choices.length?'Refresh catalog':'Load provider catalog'}</button></div>
      {choices.length===0 ? <div className="market-form market-form-three"><label>Material<input value={material} onChange={e=>{setMaterial(e.target.value);setFilamentId('')}} maxLength={60}/></label><label>Color<input value={color} onChange={e=>{setColor(e.target.value);setFilamentId('')}} maxLength={60}/></label><label>Quantity<input type="number" min={1} max={1000} value={quantity} onChange={e=>setQuantity(Math.max(1,Math.min(1000,Number(e.target.value)||1)))}/></label></div>
      : <div className="market-form market-form-three"><label>Filter catalog<input placeholder="Material or color" value={materialSearch} onChange={e=>setMaterialSearch(e.target.value)}/></label><label>Material & color<select value={filamentId} onChange={e=>{const chosen=choices.find(x=>x.id===e.target.value);if(chosen){setFilamentId(chosen.id);setMaterial(chosen.material);setColor(chosen.color)}}}><option value="" disabled>Choose from catalog</option>{filteredChoices.map(x=><option value={x.id} key={x.id}>{x.material} · {x.color}</option>)}{selectedMaterial&&!filteredChoices.some(x=>x.id===selectedMaterial.id)&&<option value={selectedMaterial.id}>{selectedMaterial.material} · {selectedMaterial.color}</option>}</select></label><label>Quantity<input type="number" min={1} max={1000} value={quantity} onChange={e=>setQuantity(Math.max(1,Math.min(1000,Number(e.target.value)||1)))}/></label></div>}
      {selectedMaterial&&<div className="market-chosen">{selectedMaterial.hex&&<span className="market-color-swatch" style={{backgroundColor:selectedMaterial.hex}}/>}<strong>{selectedMaterial.material} · {selectedMaterial.color}</strong><CheckCircle2 size={16}/></div>}
      <label className="market-check"><input type="checkbox" checked={uploadConsent} onChange={e=>setUploadConsent(e.target.checked)}/><span>I agree to send the current STL to Slant 3D for a live quotation. No order will be placed.</span></label>
      <button className="studio-btn accent market-primary-quote" disabled={!modelReady||!uploadConsent||!!quoteBusy} onClick={()=>void runQuote()}><RefreshCw size={16}/>{quoteBusy==='quote'?'Contacting Slant 3D…':'Request live quote'}</button>
    </div>

    {realQuote && <div className="market-quote"><div className="market-result-title"><CheckCircle2 size={17}/><div><strong>Quote response received</strong><span>Provider data · no payment or order</span></div></div><QuoteSummary value={realQuote.result}/>
      {quoteId ? <><div className="market-followup"><div><strong>Quantity pricing</strong><span>Quote ID {quoteId}</span></div><button className="studio-btn ghost" disabled={!!quoteBusy} onClick={()=>void updateQuantity()}>{quoteBusy==='quantity'?'Updating…':'Update quantity price'}</button></div>{qtyResult&&<QuoteSummary value={qtyResult.result}/>}<div className="market-shipping"><div className="market-subhead"><strong>Optional delivery estimate</strong><span>Address is sent only after separate consent.</span></div><div className="market-form market-shipping-grid">{(['name','email','line1','city','state','postal_code','country'] as const).map(key=><label key={key}>{key.replace('_',' ')}<input value={address[key]} onChange={e=>setAddress(prev=>({...prev,[key]:e.target.value}))} maxLength={220}/></label>)}<label>Carrier/service ID <span>(optional)</span><input value={shippingService} onChange={e=>setShippingService(e.target.value)} maxLength={200}/></label></div><label className="market-check"><input type="checkbox" checked={shipConsent} onChange={e=>setShipConsent(e.target.checked)}/><span>I agree to share these shipping details with Slant 3D for a delivery quote.</span></label><button className="studio-btn ghost" disabled={!shipConsent||!!quoteBusy||!address.line1||!address.city||!address.country||!address.postal_code||!address.name||!address.email} onClick={()=>void updateShipping()}>{quoteBusy==='shipping'?'Getting rates…':'Get delivery rates'}</button>{shippingResult&&<QuoteSummary value={shippingResult.result} type="shipping"/>}</div></> : <p className="market-alert">The provider did not return a recognized quote ID. Quantity and shipping requests are unavailable.</p>}
      <div className="market-stop"><ShieldCheck size={18}/><span><strong>Stops before payment.</strong> CADia does not create checkout links, charge a card, confirm an order, or start production.</span></div>
    </div>}
    {quoteError&&<p className="market-alert" role="alert"><TriangleAlert size={16}/>{quoteError}</p>}
    <p className="market-disclaimer">Slant 3D is an independent third-party provider. Materials, lead time, shipping eligibility and final pricing remain provider-controlled.</p>
  </section>
}
