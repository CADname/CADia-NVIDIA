import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Camera, FileImage, Loader2, ScanLine, Sparkles, Upload, X } from 'lucide-react'
import { api } from '../api'

type Props = {
  projectId: string
  connected: boolean
  selectedModel?: string
  effort?: string
  onClose: () => void
  onError: (reason: unknown) => void
  onComplete: () => void | Promise<void>
  onConnect: () => void
}

type NvidiaReview = {
  status?: string
  accepted?: boolean
  reason?: string
  visual_delta?: number
  cosmos_before?: { visual_agreement?: number; discrepancies?: Array<{ severity?: string; description?: string }> }
  cosmos_after?: { visual_agreement?: number; discrepancies?: Array<{ severity?: string; description?: string }> }
  nemotron?: { model?: string; should_repair?: boolean }
}

type OpenCvArtifact = { label?: string; path: string; kind?: string }
type OpenCvVision = {
  mode?: string
  score?: number
  action?: string
  source?: { opencv_version?: string; confidence?: number }
  final?: { score?: number; action?: string; threshold?: number; issues?: string[] }
  artifacts?: OpenCvArtifact[]
}

function nvidiaReviewText(review?: NvidiaReview): string {
  if (!review) return ''
  if (review.status === 'corrected') return 'NVIDIA Engineering Review ✓ corrected and re-verified'
  if (review.status === 'verified') return 'NVIDIA Engineering Review ✓ visually verified'
  if (review.status === 'reviewed_no_repair') return 'NVIDIA Engineering Review ✓ reviewed; base CAD kept'
  if (review.status === 'issues_found') return 'NVIDIA Engineering Review · discrepancy found; auto-repair disabled'
  if (review.status === 'not_configured') return 'NVIDIA Engineering Review · endpoint configuration required'
  if (review.status === 'disabled') return 'NVIDIA Engineering Review · disabled'
  if (review.status === 'preserved_not_run') return 'NVIDIA lane preserved · not run in OpenCV 5 mode'
  if (['degraded', 'repair_failed', 'repair_rejected', 'stale_review'].includes(review.status || '')) return 'NVIDIA Engineering Review · base CAD kept (fail-open)'
  return review.status ? `NVIDIA Engineering Review · ${review.status}` : ''
}

function nvidiaReviewDetails(review?: NvidiaReview): string {
  if (!review) return ''
  const issues = (review.cosmos_before?.discrepancies || [])
    .map((row) => row?.description?.trim())
    .filter((value): value is string => Boolean(value))
    .slice(0, 2)
  const before = review.cosmos_before?.visual_agreement
  const after = review.cosmos_after?.visual_agreement
  const score = typeof before === 'number'
    ? (typeof after === 'number'
      ? `Visual agreement ${Math.round(before * 100)}% → ${Math.round(after * 100)}%`
      : `Visual agreement ${Math.round(before * 100)}%`)
    : ''
  const detected = issues.length ? `Detected: ${issues.join(' · ')}` : ''
  return [detected, score].filter(Boolean).join(' · ')
}

type EngineRun = { name: string; available: boolean; attempted?: boolean; ok?: boolean; message?: string }
type Job = {
  id: string
  status: 'queued' | 'running' | 'cancelling' | 'cancelled' | 'failed' | 'done'
  progress: number
  message: string
  vision_mode?: 'standard' | 'opencv5'
  error?: string | null
  deadline_at?: number | null
  result?: {
    spec?: { part_name?: string; object_summary?: string; confidence?: number; uncertainties?: string[] }
    validation?: { brep_valid?: boolean; score?: number; hard_dimensions_passed?: boolean }
    refinements?: Array<{ pass: number; action: string; accepted?: boolean; visual_agreement?: number; issues?: string[] }>
    engines?: EngineRun[]
    nvidia_review?: NvidiaReview
    opencv_vision?: OpenCvVision
  } | null
}

const MAX_PHOTOS = 8
const POLL_MS = 1300
const POLL_RETRY_MAX_MS = 5000
const POLL_DEADLINE_GRACE_MS = 60_000

export default function PhotoCADModal({ projectId, connected, selectedModel, effort, onClose, onError, onComplete, onConnect }: Props) {
  const [files, setFiles] = useState<File[]>([])
  const [notes, setNotes] = useState('')
  const [dimensionHint, setDimensionHint] = useState('')
  const [job, setJob] = useState<Job | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const timer = useRef<number | null>(null)
  const disposed = useRef(false)
  const pollFailures = useRef(0)
  const previewRefreshed = useRef(false)
  const busy = Boolean(job && !['done', 'failed', 'cancelled'].includes(job.status))
  const canStart = connected && files.length > 0 && !busy
  const engines = job?.result?.engines || []

  const fileSummary = useMemo(() => files.map((file) => file.name).join(' · '), [files])
  useEffect(() => { disposed.current = false; return () => { disposed.current = true; if (timer.current) window.clearTimeout(timer.current) } }, [])

  function choose(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files || [])
    event.target.value = ''
    if (!selected.length) return
    setFiles((current) => {
      const merged = [...current]
      for (const file of selected) {
        if (merged.length >= MAX_PHOTOS) break
        if (!merged.some((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified)) merged.push(file)
      }
      return merged.slice(0, MAX_PHOTOS)
    })
  }

  async function poll(id: string) {
    if (disposed.current) return
    if (timer.current) { window.clearTimeout(timer.current); timer.current = null }
    try {
      const next = await api<Job>(`/api/projects/${projectId}/photo-cad/jobs/${id}`)
      if (disposed.current) return
      pollFailures.current = 0
      setJob(next)
      if (next.status === 'running' && next.progress >= 92 && !previewRefreshed.current) {
        previewRefreshed.current = true
        try { await onComplete() } catch {}
      }
      if (next.status === 'done') { await onComplete(); return }
      if (next.status === 'failed' || next.status === 'cancelled') return
      if (next.deadline_at && Date.now() > next.deadline_at * 1000 + POLL_DEADLINE_GRACE_MS) {
        setJob({ ...next, status: 'failed', message: 'Photo reconstruction timed out', error: 'The server did not reach a terminal state before the hard job deadline.' })
        try { await api<Job>(`/api/projects/${projectId}/photo-cad/jobs/${id}/cancel`, { method: 'POST' }) } catch {}
        return
      }
      timer.current = window.setTimeout(() => void poll(id), POLL_MS)
    } catch (reason) {
      if (disposed.current) return
      pollFailures.current += 1
      const retryMs = Math.min(POLL_RETRY_MAX_MS, POLL_MS * Math.max(1, pollFailures.current))
      setJob((current) => current && current.id === id ? { ...current, message: `Connection interrupted; retrying status (${pollFailures.current})…` } : current)
      // A transient GET failure must never permanently stop polling and leave the modal busy forever.
      timer.current = window.setTimeout(() => void poll(id), retryMs)
      // Transient polling failures are shown inline and retried; do not emit a huge gateway-error toast.
      void reason
    }
  }

  async function start() {
    if (!files.length) return
    if (!connected) { onConnect(); return }
    const form = new FormData()
    files.forEach((file) => form.append('photos', file))
    form.append('notes', notes)
    form.append('dimension_hint', dimensionHint)
    form.append('model', selectedModel || 'gpt-5.6-sol')
    form.append('effort', effort || 'high')
    try {
      const created = await api<Job>(`/api/projects/${projectId}/photo-cad/jobs`, { method: 'POST', body: form })
      pollFailures.current = 0
      previewRefreshed.current = false
      setJob(created)
      void poll(created.id)
    } catch (reason) { onError(reason) }
  }

  async function cancel() {
    if (!job) return
    try {
      const next = await api<Job>(`/api/projects/${projectId}/photo-cad/jobs/${job.id}/cancel`, { method: 'POST' })
      setJob(next)
      if (!['done', 'failed', 'cancelled'].includes(next.status)) void poll(job.id)
    } catch (reason) { onError(reason) }
  }

  const result = job?.result
  return (
    <div className="modal-backdrop visual-cad-backdrop" onMouseDown={(event) => { if (event.currentTarget === event.target && !busy) onClose() }}>
      <section className="visual-cad-modal">
        <header>
          <div><span className="visual-cad-icon"><Camera size={20} /></span><div><h2>Photo → CAD</h2><p>Product / physical-object photos to editable parametric B-Rep</p></div></div>
          <button className="icon-button" disabled={busy} onClick={onClose}><X size={18} /></button>
        </header>

        {!connected && <div className="visual-cad-warning">ChatGPT / Codex must be connected. <button onClick={onConnect}>Connect AI</button></div>}

        <div className="visual-cad-grid">
          {files.map((file, index) => (
            <div key={`${file.name}-${file.lastModified}-${index}`} className="visual-cad-drop has-file">
              <FileImage size={22} />
              <strong>Photo {index + 1}</strong>
              <small title={file.name}>{file.name}</small>
              {!busy && <button className="icon-button" title="Remove photo" onClick={() => setFiles((current) => current.filter((_, row) => row !== index))}><X size={14} /></button>}
            </div>
          ))}
          {files.length < MAX_PHOTOS && <label className="visual-cad-drop">
            <input ref={inputRef} type="file" multiple accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={choose} />
            <Upload size={22} />
            <strong>{files.length ? 'Add more photos' : 'Add product photos *'}</strong>
            <small>1–{MAX_PHOTOS} views of the same object</small>
          </label>}
        </div>

        <label className="visual-cad-field"><span>Known dimension / scale anchor <small>strongly recommended</small></span><input disabled={busy} value={dimensionHint} onChange={(e) => setDimensionHint(e.target.value)} placeholder="e.g. center hole Ø20 mm, overall height 85 mm" /></label>
        <label className="visual-cad-field"><span>Notes <small>optional</small></span><textarea disabled={busy} rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="e.g. same part in every photo, back is symmetric, ignore the clamp holding the part…" /></label>

        {files.length > 0 && !job && <div className="visual-cad-result"><div><strong>{files.length} photo{files.length === 1 ? '' : 's'} ready</strong><span>All views are treated as the same physical object.</span></div><small>{fileSummary}</small></div>}

        {job && <div className={`visual-cad-progress ${job.status}`}>
          <div className="visual-cad-progress-row"><span>{busy && <Loader2 className="spin" size={15} />}{job.message}</span><b>{job.progress}%</b></div>
          <div className="visual-cad-progress-track"><i style={{ width: `${job.progress}%` }} /></div>
          {job.error && <p className="visual-cad-error">{job.error}</p>}
        </div>}

        {job?.status === 'done' && result && <div className="visual-cad-result">
          <div><strong>{result.spec?.part_name || 'Reconstructed product'}</strong><span>{result.validation?.brep_valid ? 'Valid editable B-Rep' : 'Check B-Rep'} · score {Math.round((result.validation?.score || 0) * 100)}%</span></div>
          <div className="visual-cad-result-metrics"><span>{files.length} source photo{files.length === 1 ? '' : 's'}</span><span>{result.refinements?.length || 0} visual correction pass(es)</span><span>{result.validation?.hard_dimensions_passed ? 'Metric anchor satisfied' : dimensionHint ? 'Check metric anchor' : 'No hard metric anchor'}</span></div>
          {!!engines.length && <small>Optional engines: {engines.map((engine) => `${engine.name} ${engine.ok ? '✓' : engine.attempted ? 'failed' : 'not active'}`).join(' · ')}</small>}
          {!!result.spec?.uncertainties?.length && <small>Uncertain: {result.spec.uncertainties.slice(0, 3).join(' · ')}</small>}
          {result.opencv_vision && <div className="opencv-result">
            <div><ScanLine size={15} /><b>OpenCV {result.opencv_vision.source?.opencv_version || '5'} closed loop</b><span>{Math.round((result.opencv_vision.score || result.opencv_vision.final?.score || 0) * 100)}% geometric agreement · {result.opencv_vision.action || result.opencv_vision.final?.action || 'verified'}</span></div>
            {!!result.opencv_vision.final?.issues?.length && <small>{result.opencv_vision.final.issues.slice(0, 2).join(' · ')}</small>}
            {!!result.opencv_vision.artifacts?.length && <div className="opencv-artifacts">{result.opencv_vision.artifacts.filter((a) => a.kind !== 'edges').slice(0, 4).map((artifact) => <figure key={artifact.path}><img src={`/api/projects/${projectId}/photo-cad/jobs/${job.id}/opencv-artifacts/${artifact.path.split('/').map(encodeURIComponent).join('/')}`} alt={artifact.label || 'OpenCV evidence'} /><figcaption>{artifact.label || 'OpenCV evidence'}</figcaption></figure>)}</div>}
          </div>}
          {!!nvidiaReviewText(result.nvidia_review) && <small>{nvidiaReviewText(result.nvidia_review)}</small>}
          {!!nvidiaReviewDetails(result.nvidia_review) && <small>{nvidiaReviewDetails(result.nvidia_review)}</small>}
        </div>}

        <footer>
          <div className="visual-cad-footnote">Native path stays unchanged. When configured, NVIDIA Cosmos verifies source photos against CAD renders; Nebius Nemotron approves only bounded repair plans before Codex edits the existing B-Rep.</div>
          <div className="visual-cad-actions">
            {busy ? <button className="secondary" onClick={cancel}>Cancel</button> : <button className="secondary" onClick={onClose}>{job?.status === 'done' ? 'Close' : 'Cancel'}</button>}
            {job?.status !== 'done' && <button className="primary" disabled={!canStart} onClick={start}><Sparkles size={16} />Reconstruct CAD</button>}
          </div>
        </footer>
      </section>
    </div>
  )
}
