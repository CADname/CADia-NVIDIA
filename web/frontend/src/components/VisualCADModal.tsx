import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { FileImage, Loader2, Ruler, ScanLine, Sparkles, Upload, X } from 'lucide-react'
import { api } from '../api'

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
  status?: 'queued' | 'running' | 'done' | 'failed' | 'skipped_stale'
  read_only?: boolean
  score?: number
  action?: string
  raw_action?: string
  reason?: string
  issues?: string[]
  rescue?: { attempted?: boolean; trigger?: string; accepted_for_modeling?: boolean; baseline_score?: number; rescue_score?: number; reason?: string } | null
  source?: { opencv_version?: string; confidence?: number; views?: Record<string, { line_count?: number; circle_count?: number; confidence?: number }> }
  final?: { score?: number; action?: string; raw_action?: string; threshold?: number; issues?: string[] }
  performance?: { total_verify_ms?: number }
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
  if (review.status === 'disabled_for_product_pipeline') return 'NVIDIA review preserved · not on this Drawing-to-CAD path'
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

type Props = {
  projectId: string
  connected: boolean
  selectedModel?: string
  effort?: string
  onClose: () => void
  onError: (reason: unknown) => void
  onComplete: () => void | Promise<void>
  onConnect: () => void
  initialVisionMode?: 'standard' | 'opencv5'
}

type Job = {
  id: string
  status: 'queued' | 'running' | 'cancelling' | 'cancelled' | 'failed' | 'done'
  progress: number
  message: string
  vision_mode?: 'standard' | 'opencv5'
  deadline_at?: number | null
  error?: string | null
  result?: {
    spec?: { part_name?: string; confidence?: number; uncertainties?: string[] }
    partspec?: { part_name?: string; confidence?: number; uncertainties?: string[] }
    validation?: { brep_valid?: boolean; score?: number; dimension_checks?: Array<{ axis: string; expected_mm: number; actual_mm: number; pass: boolean }> }
    brep_validation?: { brep_valid?: boolean; solid_present?: boolean; score?: number }
    refinements?: Array<{ pass: number; action: string; confidence?: number; issues?: string[] }>
    nvidia_review?: NvidiaReview
    opencv_vision?: OpenCvVision
    opencv_rescue?: OpenCvVision['rescue']
  } | null
}

type Slot = 'primary' | 'top' | 'right' | 'reference'

const slotLabels: Record<Slot, string> = {
  primary: 'Main drawing / Front',
  top: 'Top view',
  right: 'Right view',
  reference: 'Extra / Isometric',
}

const POLL_MS = 1300
const POLL_RETRY_MAX_MS = 5000
const POLL_DEADLINE_GRACE_MS = 60_000

export default function VisualCADModal({ projectId, connected, selectedModel, effort, onClose, onError, onComplete, onConnect, initialVisionMode = 'standard' }: Props) {
  const [files, setFiles] = useState<Record<Slot, File | null>>({ primary: null, top: null, right: null, reference: null })
  const [notes, setNotes] = useState('')
  const [dimensionHint, setDimensionHint] = useState('')
  const [visionMode, setVisionMode] = useState<'standard' | 'opencv5'>(initialVisionMode)
  const [job, setJob] = useState<Job | null>(null)
  const timer = useRef<number | null>(null)
  const previewRefreshed = useRef(false)
  const pollFailures = useRef(0)
  const disposed = useRef(false)
  const busy = Boolean(job && !['done', 'failed', 'cancelled'].includes(job.status))
  const canStart = connected && Boolean(files.primary) && !busy

  const fileCount = useMemo(() => Object.values(files).filter(Boolean).length, [files])

  useEffect(() => () => {
    disposed.current = true
    if (timer.current) window.clearTimeout(timer.current)
  }, [])

  function choose(slot: Slot, event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] || null
    event.target.value = ''
    setFiles((current) => ({ ...current, [slot]: file }))
  }

  async function poll(id: string) {
    if (disposed.current) return
    try {
      const next = await api<Job>(`/api/projects/${projectId}/visual-cad/jobs/${id}`)
      if (disposed.current) return
      pollFailures.current = 0
      setJob(next)
      if (next.status === 'running' && next.progress >= 92 && !previewRefreshed.current) {
        previewRefreshed.current = true
        try { await onComplete() } catch {}
      }
      if (next.status === 'done') {
        if (!previewRefreshed.current) {
          previewRefreshed.current = true
          try { await onComplete() } catch {}
        }
        const inspectionStatus = next.result?.opencv_vision?.status
        const inspectionPending = next.vision_mode === 'opencv5' && (inspectionStatus === 'queued' || inspectionStatus === 'running')
        if (inspectionPending) {
          // CAD is already complete. Keep polling only for the independent read-only
          // inspection card; this does not keep the modeling UI busy.
          timer.current = window.setTimeout(() => void poll(id), POLL_MS)
          return
        }
        timer.current = null
        return
      }
      if (next.status === 'failed' || next.status === 'cancelled') {
        timer.current = null
        return
      }
      if (next.deadline_at && Date.now() > next.deadline_at * 1000 + POLL_DEADLINE_GRACE_MS) {
        timer.current = null
        setJob({ ...next, status: 'failed', message: 'Drawing reconstruction timed out', error: 'The server did not reach a terminal state before the hard job deadline.' })
        try { await api<Job>(`/api/projects/${projectId}/visual-cad/jobs/${id}/cancel`, { method: 'POST' }) } catch {}
        return
      }
      timer.current = window.setTimeout(() => void poll(id), POLL_MS)
    } catch (reason) {
      if (disposed.current) return
      pollFailures.current += 1
      const retryMs = Math.min(POLL_RETRY_MAX_MS, POLL_MS * Math.max(1, pollFailures.current))
      setJob((current) => current && current.id === id ? { ...current, message: `Connection interrupted; retrying status (${pollFailures.current})…` } : current)
      // Match Photo-to-CAD: transient status GET failures must never strand the UI in "running" forever.
      timer.current = window.setTimeout(() => void poll(id), retryMs)
      void reason
    }
  }

  async function start() {
    if (!files.primary) return
    if (!connected) { onConnect(); return }
    const form = new FormData()
    for (const slot of Object.keys(files) as Slot[]) {
      const file = files[slot]
      if (file) form.append(slot, file)
    }
    form.append('notes', notes)
    form.append('dimension_hint', dimensionHint)
    form.append('model', selectedModel || 'gpt-5.6-sol')
    form.append('effort', effort || 'high')
    form.append('vision_mode', visionMode)
    try {
      const created = await api<Job>(`/api/projects/${projectId}/visual-cad/drawing/jobs`, { method: 'POST', body: form })
      pollFailures.current = 0
      previewRefreshed.current = false
      setJob(created)
      void poll(created.id)
    } catch (reason) { onError(reason) }
  }

  async function cancel() {
    if (!job) return
    try {
      const next = await api<Job>(`/api/projects/${projectId}/visual-cad/jobs/${job.id}/cancel`, { method: 'POST' })
      setJob(next)
      if (!['done', 'failed', 'cancelled'].includes(next.status)) {
        if (timer.current) window.clearTimeout(timer.current)
        timer.current = window.setTimeout(() => void poll(job.id), 100)
      }
    } catch (reason) { onError(reason) }
  }

  const result = job?.result
  const resultSpec = result?.partspec || result?.spec
  return (
    <div className="modal-backdrop visual-cad-backdrop" onMouseDown={(event) => { if (event.currentTarget === event.target && !busy) onClose() }}>
      <section className="visual-cad-modal">
        <header>
          <div><span className="visual-cad-icon"><Ruler size={20} /></span><div><h2>Drawing → CAD</h2><p>Engineering drawing to editable parametric B-Rep</p></div></div>
          <button className="icon-button" disabled={busy} onClick={onClose}><X size={18} /></button>
        </header>

        {!connected && <div className="visual-cad-warning">ChatGPT / Codex must be connected for Phase 1. <button onClick={onConnect}>Connect AI</button></div>}

        <div className="vision-mode-switch">
          <button disabled={busy} className={visionMode === 'standard' ? 'active' : ''} onClick={() => setVisionMode('standard')}>
            <Sparkles size={16} /><span><b>Standard CADia</b><small>Fast path · Codex → PartSpec → native CADia validation</small></span>
          </button>
          <button disabled={busy} className={visionMode === 'opencv5' ? 'active opencv' : 'opencv'} onClick={() => setVisionMode('opencv5')}>
            <ScanLine size={16} /><span><b>OpenCV 5 Assist</b><small>Same fast path · failure-only rescue + async read-only inspection</small></span>
          </button>
        </div>

        <div className="visual-cad-grid">
          {(Object.keys(slotLabels) as Slot[]).map((slot) => (
            <label key={slot} className={`visual-cad-drop ${files[slot] ? 'has-file' : ''}`}>
              <input type="file" accept=".pdf,image/png,image/jpeg,image/webp" disabled={busy} onChange={(event) => choose(slot, event)} />
              {files[slot] ? <FileImage size={22} /> : <Upload size={22} />}
              <strong>{slotLabels[slot]}{slot === 'primary' ? ' *' : ''}</strong>
              <small>{files[slot]?.name || (slot === 'primary' ? 'PDF or image · single sheet or front view' : 'Optional')}</small>
            </label>
          ))}
        </div>

        <label className="visual-cad-field"><span>Known dimension / scale reference <small>optional</small></span><input disabled={busy} value={dimensionHint} onChange={(e) => setDimensionHint(e.target.value)} placeholder="e.g. overall width 100 mm, center hole Ø20 mm" /></label>
        <label className="visual-cad-field"><span>Notes <small>optional</small></span><textarea disabled={busy} rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Part intent, ambiguous section view, tolerances to prioritize…" /></label>

        {job && <div className={`visual-cad-progress ${job.status}`}>
          <div className="visual-cad-progress-row"><span>{busy && <Loader2 className="spin" size={15} />}{job.message}</span><b>{job.progress}%</b></div>
          <div className="visual-cad-progress-track"><i style={{ width: `${job.progress}%` }} /></div>
          {job.error && <p className="visual-cad-error">{job.error}</p>}
        </div>}

        {job?.status === 'done' && result && <div className="visual-cad-result">
          <div><strong>{resultSpec?.part_name || 'Reconstructed part'}</strong><span>{(result.brep_validation?.brep_valid ?? result.validation?.brep_valid) ? 'Valid B-Rep' : 'Check B-Rep'} · score {Math.round((result.validation?.score || 0) * 100)}%</span></div>
          <div className="visual-cad-result-metrics"><span>{fileCount} drawing{fileCount === 1 ? '' : 's'}</span><span>{result.refinements?.length || 0} synchronous visual repair pass(es)</span><span>{Math.round((resultSpec?.confidence || 0) * 100)}% extraction confidence</span></div>
          {!!resultSpec?.uncertainties?.length && <small>Uncertain: {resultSpec.uncertainties.slice(0, 2).join(' · ')}</small>}
          {result.opencv_vision && <div className="opencv-result">
            <div><ScanLine size={15} /><b>OpenCV {result.opencv_vision.source?.opencv_version || '5'} read-only inspection</b><span>{result.opencv_vision.status === 'done' ? `${Math.round((result.opencv_vision.score || result.opencv_vision.final?.score || 0) * 100)}% geometric agreement · ${result.opencv_vision.action || 'review'}` : result.opencv_vision.status === 'failed' ? 'inspection unavailable' : result.opencv_vision.status === 'skipped_stale' ? 'skipped after model changed' : 'running after CAD completion…'}</span></div>
            {result.opencv_vision.rescue?.attempted && <small>Failure-only rescue: {result.opencv_vision.rescue.accepted_for_modeling ? 'accepted by CADia objective validation' : 'not used'}{result.opencv_vision.rescue.reason ? ` · ${result.opencv_vision.rescue.reason}` : ''}</small>}
            {!!(result.opencv_vision.final?.issues || result.opencv_vision.issues)?.length && <small>{(result.opencv_vision.final?.issues || result.opencv_vision.issues || []).slice(0, 2).join(' · ')}</small>}
            {result.opencv_vision.reason && <small>{result.opencv_vision.reason}</small>}
            {!!result.opencv_vision.artifacts?.length && <div className="opencv-artifacts">{result.opencv_vision.artifacts.filter((a) => a.kind !== 'edges').slice(0, 4).map((artifact) => <figure key={artifact.path}><img src={`/api/projects/${projectId}/visual-cad/jobs/${job.id}/opencv-artifacts/${artifact.path.split('/').map(encodeURIComponent).join('/')}`} alt={artifact.label || 'OpenCV evidence'} /><figcaption>{artifact.label || 'OpenCV evidence'}</figcaption></figure>)}</div>}
          </div>}
          {!!nvidiaReviewText(result.nvidia_review) && <small>{nvidiaReviewText(result.nvidia_review)}</small>}
          {!!nvidiaReviewDetails(result.nvidia_review) && <small>{nvidiaReviewDetails(result.nvidia_review)}</small>}
        </div>}

        <footer>
          <div className="visual-cad-footnote">{visionMode === 'opencv5' ? 'Successful jobs use the same CADia modeling path as Standard. OpenCV runs only after a failed/low-score route for one rescue attempt, then performs a read-only inspection after CAD completion. It never auto-edits a completed model.' : 'Fast product path: Codex interprets the drawing, CADia builds the editable B-Rep, and Drawing2CAD/CADia objective validators decide acceptance. NVIDIA review is preserved in source but off this critical path.'}</div>
          <div className="visual-cad-actions">
            {busy ? <button className="secondary" onClick={cancel}>Cancel</button> : <button className="secondary" onClick={onClose}>{job?.status === 'done' ? 'Close' : 'Cancel'}</button>}
            {job?.status !== 'done' && <button className="primary" disabled={!canStart} onClick={start}><Sparkles size={16} />Reconstruct CAD</button>}
          </div>
        </footer>
      </section>
    </div>
  )
}
