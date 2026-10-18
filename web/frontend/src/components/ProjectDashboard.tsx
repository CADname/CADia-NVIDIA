import { FormEvent, useEffect, useMemo, useState } from 'react'
import { Box, CalendarDays, Check, ChevronRight, FolderKanban, LogOut, Plus, Search, Sparkles, Trash2, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { Project, User } from '../types'

type Props = { user: User; onLogout: () => void }

export default function ProjectDashboard({ user, onLogout }: Props) {
  const [projects, setProjects] = useState<Project[]>([])
  const [query, setQuery] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [error, setError] = useState('')
  const [deleteCandidate, setDeleteCandidate] = useState<Project | null>(null)
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null)
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedProjectIds, setSelectedProjectIds] = useState<Set<string>>(() => new Set())
  const [showBulkDeleteConfirm, setShowBulkDeleteConfirm] = useState(false)
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const [bulkDeleteProgress, setBulkDeleteProgress] = useState({ done: 0, total: 0 })
  const navigate = useNavigate()

  useEffect(() => {
    api<Project[]>('/api/projects').then(setProjects).catch((reason) => setError(String(reason)))
  }, [])

  async function createProject(event: FormEvent) {
    event.preventDefault()
    setError('')
    try {
      const project = await api<Project>('/api/projects', { method: 'POST', body: JSON.stringify({ name, description }) })
      setProjects((current) => [project, ...current])
      setShowCreate(false)
      setName('')
      setDescription('')
      navigate(`/projects/${project.id}`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    }
  }

  async function deleteProject() {
    if (!deleteCandidate || deletingProjectId) return
    setError('')
    setDeletingProjectId(deleteCandidate.id)
    try {
      await api<void>(`/api/projects/${deleteCandidate.id}`, { method: 'DELETE' })
      setProjects((current) => current.filter((project) => project.id !== deleteCandidate.id))
      setSelectedProjectIds((current) => {
        const next = new Set(current)
        next.delete(deleteCandidate.id)
        return next
      })
      setDeleteCandidate(null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setDeletingProjectId(null)
    }
  }

  function beginSelection() {
    setDeleteCandidate(null)
    setSelectedProjectIds(new Set())
    setSelectionMode(true)
  }

  function cancelSelection() {
    if (bulkDeleting) return
    setSelectedProjectIds(new Set())
    setShowBulkDeleteConfirm(false)
    setSelectionMode(false)
  }

  function toggleProjectSelection(projectId: string) {
    if (bulkDeleting) return
    setSelectedProjectIds((current) => {
      const next = new Set(current)
      if (next.has(projectId)) next.delete(projectId)
      else next.add(projectId)
      return next
    })
  }

  const filtered = useMemo(
    () => projects.filter((project) => `${project.name} ${project.description}`.toLowerCase().includes(query.toLowerCase())),
    [projects, query],
  )

  const allFilteredSelected = filtered.length > 0 && filtered.every((project) => selectedProjectIds.has(project.id))

  function toggleSelectAllVisible() {
    if (bulkDeleting || filtered.length === 0) return
    setSelectedProjectIds((current) => {
      const next = new Set(current)
      if (allFilteredSelected) filtered.forEach((project) => next.delete(project.id))
      else filtered.forEach((project) => next.add(project.id))
      return next
    })
  }

  async function deleteSelectedProjects() {
    if (bulkDeleting || selectedProjectIds.size === 0) return

    const ids = [...selectedProjectIds]
    const deletedIds = new Set<string>()
    const failures: string[] = []
    setError('')
    setBulkDeleting(true)
    setBulkDeleteProgress({ done: 0, total: ids.length })

    for (let index = 0; index < ids.length; index += 1) {
      const projectId = ids[index]
      try {
        await api<void>(`/api/projects/${projectId}`, { method: 'DELETE' })
        deletedIds.add(projectId)
      } catch (reason) {
        const project = projects.find((item) => item.id === projectId)
        const message = reason instanceof Error ? reason.message : String(reason)
        failures.push(`${project?.name ?? projectId}: ${message}`)
      } finally {
        setBulkDeleteProgress({ done: index + 1, total: ids.length })
      }
    }

    if (deletedIds.size > 0) {
      setProjects((current) => current.filter((project) => !deletedIds.has(project.id)))
    }

    setSelectedProjectIds(new Set(ids.filter((projectId) => !deletedIds.has(projectId))))
    setBulkDeleting(false)
    setShowBulkDeleteConfirm(false)

    if (failures.length > 0) {
      setError(`Deleted ${deletedIds.size} project${deletedIds.size === 1 ? '' : 's'}, but ${failures.length} failed: ${failures.join(' | ')}`)
    } else {
      setSelectionMode(false)
      setBulkDeleteProgress({ done: 0, total: 0 })
    }
  }

  return (
    <main className="dashboard-page">
      <header className="dashboard-header">
        <div className="brand-lockup"><span className="brand-mark">C</span><span>CAD<b>ia</b></span></div>
        <div className="header-actions"><span className="user-chip">{user.display_name.slice(0, 1)}</span><span className="user-name">{user.display_name}</span><button className="icon-button" title="Sign out" onClick={onLogout}><LogOut size={18} /></button></div>
      </header>
      <section className="dashboard-content">
        <div className="dashboard-intro">
          <div><p className="eyebrow">YOUR WORKSPACE</p><h1>What will you design?</h1><p>Each project keeps CAD documents, feature history, AI conversations, and Codex state isolated.</p></div>
          <button className="primary" onClick={() => setShowCreate(true)} disabled={selectionMode}><Plus size={17} /> New Project</button>
        </div>
        <div className={`dashboard-toolbar${selectionMode ? ' selection-active' : ''}`}>
          <div className="search-box"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search projects" /></div>
          {!selectionMode ? (
            <div className="project-toolbar-actions">
              <span>{filtered.length} projects</span>
              {projects.length > 0 && <button type="button" className="secondary compact-action" onClick={beginSelection}>Select</button>}
            </div>
          ) : (
            <div className="project-selection-toolbar">
              <span className="selection-count"><strong>{selectedProjectIds.size}</strong> selected</span>
              <button type="button" className="secondary compact-action" onClick={toggleSelectAllVisible} disabled={filtered.length === 0 || bulkDeleting}>
                {allFilteredSelected ? 'Clear visible' : 'Select all visible'}
              </button>
              <button type="button" className="danger-action compact-action" disabled={selectedProjectIds.size === 0 || bulkDeleting} onClick={() => setShowBulkDeleteConfirm(true)}>
                <Trash2 size={14} /> Delete selected
              </button>
              <button type="button" className="icon-button selection-cancel" title="Cancel selection" aria-label="Cancel selection" onClick={cancelSelection} disabled={bulkDeleting}><X size={17} /></button>
            </div>
          )}
        </div>
        {error && <div className="form-error">{error}</div>}
        <div className="project-grid">
          {!selectionMode && <button className="project-card create-card" onClick={() => setShowCreate(true)}><span className="create-icon"><Plus /></span><strong>New CAD Project</strong><small>Start from an empty parametric part</small></button>}
          {filtered.map((project, index) => {
            const selected = selectedProjectIds.has(project.id)
            return (
              <div className={`project-card project-card-shell${selected ? ' project-card-selected' : ''}`} key={project.id}>
                <button
                  className="project-card-open"
                  type="button"
                  aria-pressed={selectionMode ? selected : undefined}
                  onClick={() => selectionMode ? toggleProjectSelection(project.id) : navigate(`/projects/${project.id}`)}
                >
                  <div className={`project-visual shade-${index % 4}`}><Box size={64} strokeWidth={0.75} /><Sparkles size={17} /></div>
                  <div className="project-card-body">
                    <div className="project-title"><FolderKanban size={16} /><strong>{project.name}</strong>{!selectionMode && <ChevronRight size={16} />}</div>
                    <p>{project.description || 'No description'}</p>
                    <span><CalendarDays size={13} /> {new Date(project.updated_at).toLocaleDateString('en-US')}</span>
                  </div>
                </button>
                {selectionMode ? (
                  <button
                    className={`project-select-button${selected ? ' selected' : ''}`}
                    type="button"
                    aria-label={`${selected ? 'Deselect' : 'Select'} project ${project.name}`}
                    aria-pressed={selected}
                    disabled={bulkDeleting}
                    onClick={() => toggleProjectSelection(project.id)}
                  >
                    {selected && <Check size={16} strokeWidth={2.7} />}
                  </button>
                ) : (
                  <button
                    className="project-delete-button"
                    type="button"
                    title={`Delete ${project.name}`}
                    aria-label={`Delete project ${project.name}`}
                    disabled={deletingProjectId === project.id}
                    onClick={() => setDeleteCandidate(project)}
                  >
                    {deletingProjectId === project.id ? <span className="spinner" /> : <Trash2 size={15} />}
                  </button>
                )}
              </div>
            )
          })}
        </div>
      </section>
      {showCreate && (
        <div className="modal-backdrop" onMouseDown={() => setShowCreate(false)}>
          <form className="modal-card create-modal" onSubmit={createProject} onMouseDown={(event) => event.stopPropagation()}>
            <p className="eyebrow">NEW PROJECT</p><h2>New CAD Project</h2>
            <label>Project name<input autoFocus value={name} onChange={(event) => setName(event.target.value)} placeholder="Example: Gearbox concept" required maxLength={120} /></label>
            <label>Description<textarea value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Optional" rows={3} /></label>
            <div className="modal-actions"><button type="button" className="secondary" onClick={() => setShowCreate(false)}>Cancel</button><button className="primary">Create Project</button></div>
          </form>
        </div>
      )}
      {deleteCandidate && (
        <div className="modal-backdrop" onMouseDown={() => !deletingProjectId && setDeleteCandidate(null)}>
          <div className="modal-card delete-project-modal" role="dialog" aria-modal="true" aria-labelledby="delete-project-title" onMouseDown={(event) => event.stopPropagation()}>
            <span className="delete-project-icon"><Trash2 size={21} /></span>
            <p className="eyebrow">DELETE PROJECT</p>
            <h2 id="delete-project-title">Delete “{deleteCandidate.name}”?</h2>
            <p className="muted">This permanently deletes the project and its stored CAD files. This action cannot be undone.</p>
            <div className="modal-actions">
              <button type="button" className="secondary" disabled={!!deletingProjectId} onClick={() => setDeleteCandidate(null)}>Cancel</button>
              <button type="button" className="danger-action" disabled={!!deletingProjectId} onClick={deleteProject}>
                {deletingProjectId ? <><span className="spinner" /> Deleting…</> : <><Trash2 size={15} /> Delete Project</>}
              </button>
            </div>
          </div>
        </div>
      )}
      {showBulkDeleteConfirm && (
        <div className="modal-backdrop" onMouseDown={() => !bulkDeleting && setShowBulkDeleteConfirm(false)}>
          <div className="modal-card delete-project-modal" role="dialog" aria-modal="true" aria-labelledby="bulk-delete-project-title" onMouseDown={(event) => event.stopPropagation()}>
            <span className="delete-project-icon"><Trash2 size={21} /></span>
            <p className="eyebrow">DELETE SELECTED PROJECTS</p>
            <h2 id="bulk-delete-project-title">Delete {selectedProjectIds.size} selected project{selectedProjectIds.size === 1 ? '' : 's'}?</h2>
            <p className="muted">The selected projects and their stored CAD files will be permanently deleted. This action cannot be undone.</p>
            {bulkDeleting && <div className="bulk-delete-progress"><span className="spinner" /><span>Deleting {bulkDeleteProgress.done} of {bulkDeleteProgress.total}…</span></div>}
            <div className="modal-actions">
              <button type="button" className="secondary" disabled={bulkDeleting} onClick={() => setShowBulkDeleteConfirm(false)}>Cancel</button>
              <button type="button" className="danger-action" disabled={bulkDeleting || selectedProjectIds.size === 0} onClick={deleteSelectedProjects}>
                {bulkDeleting ? <><span className="spinner" /> Deleting…</> : <><Trash2 size={15} /> Delete {selectedProjectIds.size}</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
