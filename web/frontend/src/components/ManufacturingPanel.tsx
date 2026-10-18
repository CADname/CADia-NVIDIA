import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ArrowLeft, ArrowRight, Box, Check, CheckCircle2, ChevronDown, ClipboardCheck,
  Download, ExternalLink, Factory, FileCheck2, FileDown, ListChecks,
  MapPin, Package, Plus, Printer, RefreshCw, Send, ShieldCheck,
  ShoppingBag, TriangleAlert, Truck, Wrench, X,
} from 'lucide-react'
import { api, download } from '../api'
import '../studio.css'
import ManufacturingMarketplace from './ManufacturingMarketplace'

export type ManufacturingCapabilities = {
  native: { dfm_mesh_precheck: boolean; three_mf_core: boolean; step_ap242: boolean }
  optional: {
    lib3mf: Record<string, unknown>
    slicer: { backend?: string; available?: boolean; detail?: string; license_boundary?: string }
    cam: { available?: boolean; library?: string; license?: string; detail?: string }
  }
  printers: Array<{ name: string; kind: string; configured: boolean; execution_enabled: boolean }>
  safety: string
}
type DfmReport = {
  ok: boolean; scope: string
  checks: Array<{ id: string; status: 'pass' | 'warning' | 'fail'; message: string; value?: unknown }>
  limitations?: string[]
}
type StudioOrder = {
  id: string; project_id: string; model_name: string; model_revision: number
  process: string; material: string; color: string; quantity: number; finish: string
  delivery: 'shipping' | 'pickup'; notes: string; customer_name: string; customer_email: string
  phone?: string; shipping_address?: string
  status: string; quote_krw: number | null; quote_note: string; eta_days: number | null
  shipping_carrier: string; tracking_number: string; created_at: string; updated_at: string
}
type Tab = 'create' | 'quote' | 'orders' | 'tools' | 'admin'
type Props = {
  projectId: string; onClose: () => void; onError: (reason: unknown) => void
  modelName?: string; modelReady?: boolean; revision?: number; dimensions?: [number, number, number] | null
  customerEmail?: string; customerName?: string
}

const materials = [
  { id: 'pla', title: 'PLA', detail: 'Everyday prototypes', subtitle: 'Easy, clean finish' },
  { id: 'petg', title: 'PETG', detail: 'Functional parts', subtitle: 'Tougher, flexible' },
  { id: 'abs', title: 'ABS', detail: 'Heat-resistant parts', subtitle: 'Quote-dependent' },
  { id: 'tpu', title: 'TPU', detail: 'Flexible parts', subtitle: 'Quote-dependent' },
]
const colors = [
  { id: 'black', label: 'Black', css: '#202229' }, { id: 'white', label: 'White', css: '#f4f6f8' },
  { id: 'gray', label: 'Gray', css: '#8d949d' }, { id: 'blue', label: 'Blue', css: '#4e8ce7' },
  { id: 'red', label: 'Red', css: '#d96b68' }, { id: 'custom', label: 'Ask us', css: 'linear-gradient(120deg,#4f88f9,#d06abb,#e4c176)' },
]
const steps = ['Configure', 'Contact', 'Review'] as const
const statusLabels: Record<string, string> = {
  requested: 'Review pending', quoted: 'Quote ready', confirmed: 'Confirmed', printing: 'In production',
  quality_check: 'Quality check', shipped: 'On its way', completed: 'Complete', cancelled: 'Cancelled', rejected: 'Unavailable',
}
const money = (n: number) => `${n.toLocaleString('ko-KR')} KRW`
const niceDate = (s: string) => new Date(s.endsWith('Z') || /[+-]\d\d:\d\d$/.test(s) ? s : `${s}Z`).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })

async function postDownload(path: string, body: unknown, fallback: string) {
  const response = await fetch(path, {
    method: 'POST', credentials: 'include', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string }
    throw Error(payload.detail || `Request failed (${response.status})`)
  }
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url; anchor.download = fallback; document.body.appendChild(anchor); anchor.click(); anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1500)
}

export default function ManufacturingPanel({ projectId, onClose, modelName, modelReady, revision, dimensions, customerEmail = '', customerName = '' }: Props) {
  const [tab, setTab] = useState<Tab>('create')
  const [step, setStep] = useState(0)
  const [material, setMaterial] = useState('petg')
  const [color, setColor] = useState('black')
  const [quantity, setQuantity] = useState(1)
  const [finish, setFinish] = useState('standard')
  const [delivery, setDelivery] = useState<'shipping' | 'pickup'>('shipping')
  const [name, setName] = useState(customerName)
  const [email, setEmail] = useState(customerEmail)
  const [notes, setNotes] = useState('')
  const [consent, setConsent] = useState(false)
  const [capabilities, setCapabilities] = useState<ManufacturingCapabilities | null>(null)
  const [dfm, setDfm] = useState<DfmReport | null>(null)
  const [orders, setOrders] = useState<StudioOrder[]>([])
  const [adminOrders, setAdminOrders] = useState<StudioOrder[]>([])
  const [admin, setAdmin] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [activeOrder, setActiveOrder] = useState<string | null>(null)
  const [phone, setPhone] = useState('')
  const [address, setAddress] = useState('')
  const [quote, setQuote] = useState('')
  const [eta, setEta] = useState('5')
  const [quoteNote, setQuoteNote] = useState('')
  const [carrier, setCarrier] = useState('')
  const [tracking, setTracking] = useState('')
  const [submittedId, setSubmittedId] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'active' | 'closed'>('all')
  const printEndpoint = `/api/projects/${projectId}/manufacturing`

  const refreshOrders = useCallback(async () => setOrders(await api<StudioOrder[]>('/api/studio/orders')), [])
  const refreshAdmin = useCallback(async () => setAdminOrders(await api<StudioOrder[]>('/api/studio/admin/orders')), [])
  useEffect(() => {
    let alive = true
    api<ManufacturingCapabilities>(`${printEndpoint}/capabilities`).then(v => { if (alive) setCapabilities(v) }).catch(() => {})
    api<{ admin: boolean }>('/api/studio/meta').then(v => { if (alive) setAdmin(v.admin) }).catch(() => {})
    refreshOrders().catch(() => {})
    return () => { alive = false }
  }, [printEndpoint, refreshOrders])
  useEffect(() => { if (tab === 'admin' && admin) refreshAdmin().catch(e => setError(String(e))) }, [tab, admin, refreshAdmin])
  useEffect(() => { setError(''); setSuccess('') }, [tab, step])

  async function act(label: string, task: () => Promise<unknown>, refresh = true) {
    setBusy(label); setError(''); setSuccess('')
    try {
      await task()
      if (refresh) { await refreshOrders(); if (admin) await refreshAdmin() }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally { setBusy('') }
  }

  async function submit() {
    await act('submit', async () => {
      const response = await api<StudioOrder>(`${printEndpoint}/studio/request`, {
        method: 'POST', body: JSON.stringify({ material, color, quantity, finish, delivery, customer_name: name.trim(), customer_email: email.trim(), notes: notes.trim(), consent }),
      })
      setSubmittedId(response.id)
      setActiveOrder(response.id)
      setTab('orders')
    })
  }
  async function runDfm() {
    await act('dfm', async () => setDfm(await api<DfmReport>(`${printEndpoint}/dfm`)), false)
  }
  async function slice() {
    await act('slice', () => postDownload(`${printEndpoint}/slice`, { overrides: {} }, 'CADia-model.gcode'), false)
  }
  async function upload(target: string) {
    await act(`printer:${target}`, async () => {
      await api(`${printEndpoint}/slice-upload/${encodeURIComponent(target)}`, { method: 'POST', body: JSON.stringify({ overrides: {}, start: false }) })
      setSuccess(`G-code uploaded to ${target}. The printer was not started.`)
    }, false)
  }
  const dimensionsText = dimensions ? dimensions.map(n => n.toFixed(1)).join(' × ') + ' mm' : 'Dimensions unavailable'
  const pending = Boolean(busy)
  const canMoveContact = modelReady && quantity >= 1 && quantity <= 100
  const canMoveReview = name.trim().length >= 2 && /^\S+@\S+\.\S+$/.test(email.trim()) && consent
  const displayedOrders = useMemo(() => orders.filter(o => filter === 'all' || (filter === 'active' ? !['cancelled','completed','rejected'].includes(o.status) : ['cancelled','completed','rejected'].includes(o.status))), [orders, filter])

  return <div className="modal-backdrop cadia-studio-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
    <section className="modal-card manufacturing-modal cadia-studio" role="dialog" aria-modal="true" aria-label="CADia manufacturing studio">
      <header className="studio-header">
        <div className="studio-brand"><div className="studio-brand-icon"><Factory size={20}/></div><div><span className="studio-kicker">CADia / MANUFACTURING</span><h2>Make it real<span className="studio-header-dot">.</span></h2><p>From your CAD model to a physical part.</p></div></div>
        <button className="studio-close" onClick={onClose} aria-label="Close manufacturing studio"><X size={19}/></button>
      </header>
      <nav className="studio-tabs studio-tabs-clean" aria-label="Manufacturing sections">
        <button className={`studio-primary-tab ${tab === 'create' ? 'current' : ''}`} onClick={() => setTab('create')}><Printer size={17}/><span>Print</span></button>
        <button className={tab === 'quote' ? 'current' : ''} onClick={() => setTab('quote')}><Package size={16}/><span>Instant quote</span></button>
        <button className={tab === 'tools' ? 'current' : ''} onClick={() => setTab('tools')}><Wrench size={16}/><span>Tools</span></button>
        <button className={tab === 'orders' ? 'current' : ''} onClick={() => setTab('orders')}><ShoppingBag size={16}/><span>Requests</span>{orders.length > 0 && <i>{orders.length}</i>}</button>
        {admin && <button className={tab === 'admin' ? 'current' : ''} onClick={() => setTab('admin')}><ListChecks size={16}/><span>Studio desk</span></button>}
      </nav>
      <div className="studio-scroll">
        {tab === 'create' && <>
          <section className="studio-intro"><div><span className="studio-eyebrow"><span className="studio-live-dot"/> CADia Studio · DIRECT 3D PRINTING</span><h3>Design it. We'll make it.</h3><p>Send your CADia model for a manually reviewed 3D printing quote. Confirm the quote before any production begins.</p><div className="studio-intro-badges"><span><CheckCircle2 size={14}/> Your CAD model, connected</span><span><ShieldCheck size={14}/> Quote before production</span></div></div><div className="studio-hero-object" aria-hidden="true"><Box size={72} strokeWidth={1.05}/><span>DESIGN → PART</span></div></section>
          <div className="studio-progress">{steps.map((s,i) => <button key={s} disabled={i > step} className={step === i ? 'is-active' : i < step ? 'is-complete' : ''} onClick={() => setStep(i)}><span>{i < step ? <Check size={14}/> : i + 1}</span><strong>{s}</strong></button>)}</div>
          <div className="studio-body">
            <div className="studio-main">
              {step === 0 && <div className="studio-stack">
                <div className="studio-section-title"><span>01</span><div><h4>Printing specifications</h4><p>Set up your prototype in a few clicks.</p></div></div>
                <div className="studio-fieldgroup"><div className="studio-field-title">Manufacturing process <span>Currently available</span></div><div className="studio-process-choice"><div className="studio-process-icon"><Printer size={25}/></div><div><strong>FDM 3D printing</strong><small>Layer-by-layer plastic printing · manually reviewed</small></div><CheckCircle2 size={18}/></div></div>
                <div className="studio-fieldgroup"><div className="studio-field-title">Choose material</div><div className="studio-material-grid">{materials.map(m => <button key={m.id} className={`studio-material ${material === m.id ? 'selected' : ''}`} onClick={() => setMaterial(m.id)} aria-pressed={material === m.id}><span className="studio-card-top"><strong>{m.title}</strong>{material === m.id && <CheckCircle2 size={17}/>}</span><span>{m.detail}</span><small>{m.subtitle}</small></button>)}</div><small className="studio-hint">Material and color availability are confirmed when your quote is prepared.</small></div>
                <div className="studio-fieldgroup"><div className="studio-field-title">Color preference</div><div className="studio-colors">{colors.map(c => <button key={c.id} className={color === c.id ? 'chosen' : ''} title={c.label} aria-label={c.label} aria-pressed={color === c.id} onClick={() => setColor(c.id)}><span style={{background:c.css}}/>{color === c.id && <Check size={13}/>}<small>{c.label}</small></button>)}</div></div>
                <div className="studio-form-row"><label className="studio-label">Quantity<div className="studio-stepper"><button type="button" disabled={quantity <= 1} onClick={() => setQuantity(n => Math.max(1,n-1))} aria-label="Decrease quantity">−</button><input type="number" min={1} max={100} value={quantity} onChange={e => setQuantity(Number(e.target.value))}/><button type="button" disabled={quantity >= 100} onClick={() => setQuantity(n => Math.min(100,n+1))} aria-label="Increase quantity">+</button></div></label><label className="studio-label">Finishing<select value={finish} onChange={e => setFinish(e.target.value)}><option value="standard">Standard finish</option><option value="support-removal">Remove supports</option><option value="surface-finish">Extra surface finishing</option></select></label></div>
              </div>}
              {step === 1 && <div className="studio-stack"><div className="studio-section-title"><span>02</span><div><h4>Where should it go?</h4><p>Tell us how to reach you. Address is collected only after you accept the quote.</p></div></div>
                <div className="studio-fieldgroup"><div className="studio-field-title">Receiving method</div><div className="studio-delivery-grid"><button className={delivery === 'shipping' ? 'selected' : ''} onClick={() => setDelivery('shipping')}><Truck size={24}/><strong>Ship to me</strong><small>Delivery cost included in manual quote</small></button><button className={delivery === 'pickup' ? 'selected' : ''} onClick={() => setDelivery('pickup')}><MapPin size={24}/><strong>Local pickup</strong><small>Pickup details confirmed by the studio</small></button></div></div>
                <div className="studio-form-row"><label className="studio-label">Contact name <em>*</em><input autoComplete="name" maxLength={100} placeholder="Full name" value={name} onChange={e => setName(e.target.value)}/></label><label className="studio-label">Email <em>*</em><input type="email" autoComplete="email" maxLength={320} placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)}/></label></div>
                <label className="studio-label">Anything we should know?<textarea rows={4} maxLength={1200} value={notes} onChange={e => setNotes(e.target.value)} placeholder="Critical dimensions, intended use, tolerances, finishing preferences…"/><small className="studio-counter">{notes.length}/1200</small></label>
                <label className="studio-consent"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)}/><span>I agree to share the submitted model snapshot and contact details with the CADia Studio operator for quote handling. No payment will be taken.</span></label>
              </div>}
              {step === 2 && <div className="studio-stack"><div className="studio-section-title"><span>03</span><div><h4>Review your request</h4><p>Nothing is charged or printed when you submit.</p></div></div><div className="studio-review-grid"><div><small>Process</small><strong>FDM 3D printing</strong></div><div><small>Material</small><strong>{material.toUpperCase()}</strong></div><div><small>Color</small><strong>{colors.find(c=>c.id === color)?.label}</strong></div><div><small>Quantity</small><strong>{quantity} {quantity === 1 ? 'piece' : 'pieces'}</strong></div><div><small>Finishing</small><strong>{finish.replaceAll('-', ' ')}</strong></div><div><small>Fulfillment</small><strong>{delivery === 'shipping' ? 'Shipping' : 'Pickup'}</strong></div><div><small>Contact</small><strong>{name}</strong><small>{email}</small></div></div><div className="studio-information"><ClipboardCheck size={20}/><div><strong>What happens next?</strong><p>We'll check the model and send a price and estimated lead time in CADia. You decide whether to confirm. Printing starts only after that confirmation and operator review.</p></div></div></div>}
            </div>
            <aside className="studio-side"><div className="studio-preview"><div className="studio-preview-top"><span>CONNECTED MODEL</span><CheckCircle2 size={16}/></div><div className="studio-preview-symbol"><Box size={61} strokeWidth={1}/><div className="studio-preview-axis">X · Y · Z</div></div><strong title={modelName}>{modelName || 'Untitled CAD model'}</strong><small>{dimensionsText}</small><div className="studio-preview-meta"><span>Revision {revision ?? '—'}</span><span>{modelReady ? 'Geometry ready' : 'No geometry'}</span></div></div>
              <div className="studio-estimate"><span>Estimated price</span><strong>Custom quote</strong><small>Confirmed after reviewing the actual model, material and quantity.</small></div>
              <button className="studio-dfm-button" disabled={pending || !modelReady} onClick={runDfm}><ShieldCheck size={17}/>{busy === 'dfm' ? 'Checking model…' : 'Run printability precheck'}<ArrowRight size={16}/></button>
              {dfm && <div className={`studio-dfm-box ${dfm.ok ? 'pass' : 'fail'}`}><strong>{dfm.ok ? 'Precheck completed' : 'Model needs attention'}</strong>{dfm.checks.map(c => <p key={c.id}><span>{c.status === 'pass' ? '✓' : '!'}</span>{c.message}</p>)}<small>Automated precheck only; operator review is still required.</small></div>}
            </aside>
          </div>
          <div className="studio-footer"><div><ShieldCheck size={15}/> Manual quote · No automatic payment · No machine start</div><div className="studio-footer-actions">{step > 0 && <button className="studio-btn ghost" onClick={() => setStep(s => s-1)}><ArrowLeft size={15}/> Back</button>}{step < 2 ? <button className="studio-btn accent" disabled={step === 0 ? !canMoveContact : !canMoveReview} onClick={() => setStep(s => s+1)}>Continue <ArrowRight size={17}/></button> : <button className="studio-btn accent" disabled={pending || !modelReady} onClick={submit}>{busy === 'submit' ? 'Submitting…' : 'Request a quote'} <Send size={16}/></button>}</div></div>
        </>}
        {tab === 'orders' && <section className="studio-pane"><div className="studio-pane-head"><div><span className="studio-eyebrow">YOUR REQUESTS</span><h3>Production timeline</h3><p>Live request status from CADia. No payments are processed here.</p></div><button className="studio-btn ghost" onClick={() => act('refresh', refreshOrders, false)} disabled={pending}><RefreshCw size={15}/> Refresh</button></div>
          {submittedId && <div className="studio-success"><CheckCircle2 size={19}/><div><strong>Quote request submitted</strong><p>Your model snapshot is saved as request #{submittedId.slice(0,8).toUpperCase()}. Check back here for an operator quote.</p></div><button onClick={() => setSubmittedId(null)} aria-label="Dismiss"><X size={16}/></button></div>}
          <div className="studio-filters">{(['all','active','closed'] as const).map(f => <button key={f} className={filter === f ? 'selected' : ''} onClick={() => setFilter(f)}>{f === 'all' ? 'All requests' : f === 'active' ? 'In progress' : 'Closed'}</button>)}</div>
          {displayedOrders.length === 0 ? <div className="studio-empty"><Package size={33}/><h4>No requests to show</h4><p>Submit your first 3D print request from the current CAD project.</p><button className="studio-btn accent" onClick={() => setTab('create')}><Plus size={16}/> New print request</button></div> : <div className="studio-order-list">{displayedOrders.map(order => <article className="studio-order-card" key={order.id}><button className="studio-order-heading" onClick={() => setActiveOrder(activeOrder === order.id ? null : order.id)}><span className="studio-order-icon"><Package size={19}/></span><span className="studio-order-name"><strong>{order.model_name}</strong><small>#{order.id.slice(0,8).toUpperCase()} · {niceDate(order.created_at)}</small></span><span className={`studio-status status-${order.status}`}>{statusLabels[order.status] || order.status}</span><ChevronDown size={16}/></button>{activeOrder === order.id && <div className="studio-order-details"><div className="studio-order-facts"><div><small>Specification</small><strong>{order.material.toUpperCase()} · {order.color} · {order.quantity} pcs</strong></div><div><small>Fulfillment</small><strong>{order.delivery === 'shipping' ? 'Shipping' : 'Pickup'}</strong></div><div><small>Quoted price</small><strong>{order.quote_krw == null ? 'Awaiting quote' : money(order.quote_krw)}</strong></div><div><small>Lead time</small><strong>{order.eta_days ? `${order.eta_days} days (estimate)` : 'Not set'}</strong></div></div>{order.quote_note && <p className="studio-order-note">{order.quote_note}</p>}{order.tracking_number && <p className="studio-order-note">Shipping: {order.shipping_carrier} · {order.tracking_number}</p>}
                {order.status === 'quoted' && <div className="studio-confirm"><strong>Ready to proceed?</strong><p>Confirming accepts the quote and shares your delivery details with the operator. It does not collect payment.</p><div className="studio-form-row"><label className="studio-label">Phone number<input type="tel" maxLength={40} value={phone} onChange={e => setPhone(e.target.value)} placeholder="Contact number"/></label>{order.delivery === 'shipping' && <label className="studio-label">Shipping address<input maxLength={500} value={address} onChange={e => setAddress(e.target.value)} placeholder="Street, building, postal code"/></label>}</div><button className="studio-btn accent" disabled={pending || phone.trim().length < 7 || (order.delivery === 'shipping' && address.trim().length < 8)} onClick={() => act('confirm', () => api(`/api/studio/orders/${order.id}/confirm`, { method:'POST', body:JSON.stringify({ phone, shipping_address:address }) }))}>Accept quote and confirm <Check size={15}/></button></div>}
                {['requested','quoted'].includes(order.status) && <button className="studio-btn ghost mini" disabled={pending} onClick={() => { if (window.confirm('Cancel this print request?')) void act('cancel', () => api(`/api/studio/orders/${order.id}/cancel`, {method:'POST'})) }}>Cancel request</button>}
                <div className="studio-order-footnote">Model revision {order.model_revision} · Snapshot stored at submission</div></div>}</article>)}</div>}
        </section>}
        {tab === 'quote' && <ManufacturingMarketplace projectId={projectId} modelReady={modelReady} mode="quote"/> }
        {tab === 'tools' && <section className="studio-pane"><div className="studio-pane-head"><div><span className="studio-eyebrow">BUILT-IN TOOLS</span><h3>Prepare for manufacturing</h3><p>All original CADia export, DFM, slicing and printer tools are preserved.</p></div></div><div className="studio-tool-grid"><article className="studio-tool-card"><FileDown size={23}/><h4>Manufacturing files</h4><p>Export from your CAD document. STEP AP242 keeps B-Rep geometry; STL and 3MF are mesh-based.</p><div><button className="studio-btn ghost" onClick={() => download(`${printEndpoint}/export/step-ap242`)}><Download size={14}/> STEP AP242</button><button className="studio-btn ghost" onClick={() => download(`/api/projects/${projectId}/cad/export/stl`)}><Download size={14}/> STL</button><button className="studio-btn ghost" onClick={() => download(`${printEndpoint}/export/3mf`)}><Download size={14}/> 3MF</button></div></article><article className="studio-tool-card"><ShieldCheck size={23}/><h4>Print DFM precheck</h4><p>Check mesh closure, non-manifold edges, build volume, and basic print risks.</p><div><button className="studio-btn ghost" disabled={pending} onClick={runDfm}><RefreshCw size={14}/> {busy === 'dfm' ? 'Checking…' : 'Run precheck'}</button></div></article><article className="studio-tool-card"><FileCheck2 size={23}/><h4>Generate G-code</h4><p>{capabilities?.optional.slicer.available ? `Slicer: ${capabilities.optional.slicer.backend || 'configured'}` : capabilities?.optional.slicer.detail || 'Requires a configured slicer profile.'}</p><div><button className="studio-btn ghost" disabled={pending || !capabilities?.optional.slicer.available} onClick={slice}><Download size={14}/> {busy === 'slice' ? 'Slicing…' : 'Download G-code'}</button></div></article><article className="studio-tool-card"><Printer size={23}/><h4>Send to a printer</h4><p>Uploads G-code to a configured printer. Does not start the machine automatically.</p><div>{capabilities?.printers.length ? capabilities.printers.map(p => <button key={p.name} className="studio-btn ghost" disabled={pending || !capabilities?.optional.slicer.available} onClick={() => upload(p.name)}><Printer size={14}/> {busy === `printer:${p.name}` ? 'Uploading…' : p.name}</button>) : <span className="studio-unavailable">No printer target configured</span>}</div></article></div>{dfm && <div className="studio-tool-report"><h4>{dfm.ok ? 'DFM precheck results' : 'Review these warnings'}</h4>{dfm.checks.map(c => <p key={c.id}><span className={`dfm-${c.status}`}>{c.status}</span>{c.message}</p>)}<small>Results are preliminary, not a manufacturing guarantee.</small></div>}<div className="studio-small-note"><ShieldCheck size={16}/>{capabilities?.safety || 'Automatic machine execution is disabled.'}</div>
</section>}
        {tab === 'admin' && admin && <section className="studio-pane"><div className="studio-pane-head"><div><span className="studio-eyebrow">PRIVATE OPERATIONS</span><h3>Studio order desk</h3><p>Quote, download the submitted STL snapshot, and update production stages.</p></div><button className="studio-btn ghost" disabled={pending} onClick={() => act('adminRefresh', refreshAdmin, false)}><RefreshCw size={15}/> Refresh</button></div><div className="studio-admin-summary">{[{label:'New requests',value:adminOrders.filter(o=>o.status==='requested').length},{label:'Awaiting customer',value:adminOrders.filter(o=>o.status==='quoted').length},{label:'Production queue',value:adminOrders.filter(o=>['confirmed','printing','quality_check'].includes(o.status)).length}].map(s=><div key={s.label}><strong>{s.value}</strong><small>{s.label}</small></div>)}</div>
          {adminOrders.length === 0 && <div className="studio-empty"><Package size={30}/><h4>No requests received</h4></div>}
          <div className="studio-order-list">{adminOrders.map(o => <article className="studio-order-card" key={o.id}><button className="studio-order-heading" onClick={() => { setActiveOrder(activeOrder === o.id ? null : o.id); setQuote(o.quote_krw == null ? '' : String(o.quote_krw)); setEta(String(o.eta_days || 5)); setQuoteNote(o.quote_note || ''); setCarrier(o.shipping_carrier || ''); setTracking(o.tracking_number || '') }}><span className="studio-order-icon"><Package size={19}/></span><span className="studio-order-name"><strong>{o.model_name}</strong><small>#{o.id.slice(0,8).toUpperCase()} · {o.customer_name} · {o.customer_email}</small></span><span className={`studio-status status-${o.status}`}>{statusLabels[o.status] || o.status}</span><ChevronDown size={16}/></button>{activeOrder === o.id && <div className="studio-order-details"><div className="studio-order-facts"><div><small>Material / color</small><strong>{o.material.toUpperCase()} · {o.color}</strong></div><div><small>Quantity / finish</small><strong>{o.quantity} · {o.finish}</strong></div><div><small>Delivery</small><strong>{o.delivery}</strong></div><div><small>Contact</small><strong>{o.phone || 'Not confirmed yet'}</strong></div></div>{o.notes && <p className="studio-order-note">Customer notes: {o.notes}</p>}{o.shipping_address && <p className="studio-order-note">Shipping address: {o.shipping_address}</p>}<button className="studio-btn ghost mini" onClick={() => download(`/api/studio/admin/orders/${o.id}/model`)}><Download size={14}/> Download submitted STL</button>
                  {['requested','quoted'].includes(o.status) && <div className="studio-admin-form"><strong>Prepare manual quote</strong><div className="studio-form-row"><label className="studio-label">Total quote (KRW)<input type="number" min={0} max={100000000} value={quote} onChange={e => setQuote(e.target.value)} placeholder="e.g. 18000"/></label><label className="studio-label">Estimated days<input type="number" min={1} max={365} value={eta} onChange={e => setEta(e.target.value)}/></label></div><label className="studio-label">Message to customer<textarea value={quoteNote} maxLength={1500} onChange={e => setQuoteNote(e.target.value)} rows={2} placeholder="Materials, finish, expected schedule…"/></label><button className="studio-btn accent" disabled={pending || quote.trim()==='' || !Number.isInteger(Number(quote)) || Number(quote)<0 || Number(quote)>100000000 || !Number.isInteger(Number(eta)) || Number(eta)<1 || Number(eta)>365} onClick={() => act('quote', () => api(`/api/studio/admin/orders/${o.id}/quote`, {method:'POST',body:JSON.stringify({quote_krw:Number(quote),eta_days:Number(eta),quote_note:quoteNote})}))}>Send quote <Send size={15}/></button></div>}
                  {['confirmed','printing','quality_check','shipped'].includes(o.status) && <div className="studio-admin-form"><strong>Production management</strong>{o.status === 'quality_check' && o.delivery === 'shipping' && <div className="studio-form-row"><label className="studio-label">Carrier<input value={carrier} maxLength={80} onChange={e => setCarrier(e.target.value)} placeholder="Courier name"/></label><label className="studio-label">Tracking number<input value={tracking} maxLength={120} onChange={e => setTracking(e.target.value)} placeholder="Required for shipping"/></label></div>}<div className="studio-admin-actions">{(o.status === 'confirmed' ? [{id:'printing',label:'Start printing'}] : o.status === 'printing' ? [{id:'quality_check',label:'Mark QC'}] : o.status === 'quality_check' ? [{id:'shipped',label:'Mark shipped'},{id:'completed',label:'Mark collected / complete'}] : [{id:'completed',label:'Mark delivered'}]).filter(a => a.id !== 'shipped' || o.delivery === 'shipping').map(a => <button key={a.id} className="studio-btn accent" disabled={pending || (a.id === 'shipped' && !tracking.trim())} onClick={() => act(`stage-${a.id}`, () => api(`/api/studio/admin/orders/${o.id}/status`, {method:'PATCH',body:JSON.stringify({status:a.id,shipping_carrier:carrier,tracking_number:tracking})}))}>{a.label} <ArrowRight size={15}/></button>)}</div></div>}
                  {['requested','quoted','confirmed'].includes(o.status) && <button className="studio-btn ghost mini" disabled={pending} onClick={() => { if (window.confirm('Reject this request?')) void act('reject', () => api(`/api/studio/admin/orders/${o.id}/status`, {method:'PATCH',body:JSON.stringify({status:'rejected'})})) }}>Reject request</button>}
                </div>}</article>)}</div></section>}
        {error && <div className="studio-error" role="alert"><TriangleAlert size={17}/><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={15}/></button></div>}
        {success && <div className="studio-success"><CheckCircle2 size={17}/>{success}</div>}
      </div>
    </section>
  </div>
}
