import { useEffect, useMemo, useState } from 'react'
import { Check, Clipboard, Download, KeyRound, Network, Plus, Trash2, X } from 'lucide-react'
import { api, download } from '../api'

type TokenRow = {
  id: string
  label: string
  createdAt: string | null
  expiresAt: string | null
  lastUsedAt: string | null
  revokedAt: string | null
}

type CreatedToken = TokenRow & {
  token: string
  server: string
  rpcUrl: string
  remoteRpcUrl?: string
  bridgeDownload: string
  warning: string
}

type Props = { projectId: string; onClose: () => void }

export default function McpConnectModal({ projectId, onClose }: Props) {
  const [tokens, setTokens] = useState<TokenRow[]>([])
  const [created, setCreated] = useState<CreatedToken | null>(null)
  const [label, setLabel] = useState('My AI connection')
  const [days, setDays] = useState(30)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState('')

  async function load() {
    try { setTokens(await api<TokenRow[]>(`/api/projects/${projectId}/mcp/tokens`)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  useEffect(() => { void load() }, [projectId])

  async function create() {
    setBusy(true); setError('')
    try {
      const value = await api<CreatedToken>(`/api/projects/${projectId}/mcp/tokens`, {
        method: 'POST', body: JSON.stringify({ label, expires_days: days }),
      })
      setCreated(value); await load()
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setBusy(false) }
  }

  async function revoke(id: string) {
    try { await api<void>(`/api/projects/${projectId}/mcp/tokens/${id}`, { method: 'DELETE' }); await load() }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  async function copy(name: string, value: string) {
    try { await navigator.clipboard.writeText(value); setCopied(name); window.setTimeout(() => setCopied(''), 1200) }
    catch { setError('Could not copy to clipboard.') }
  }

  const publicOrigin = typeof window === 'undefined' ? (created?.server || '') : window.location.origin
  const remoteUrl = useMemo(() => created ? `${publicOrigin.replace(/\/$/, '')}/api/mcp` : '', [created, publicOrigin])
  const authHeader = useMemo(() => created ? `Bearer ${created.token}` : '', [created])
  const cursorConfig = useMemo(() => created ? JSON.stringify({
    mcpServers: {
      cadia: {
        type: 'streamable-http',
        url: remoteUrl,
        headers: { Authorization: authHeader },
      },
    },
  }, null, 2) : '', [created, remoteUrl, authHeader])
  const codexConfig = useMemo(() => created ? `[mcp_servers.cadia]\nurl = "${remoteUrl}"\nbearer_token_env_var = "CADIA_MCP_TOKEN"` : '', [created, remoteUrl])
  const legacyCommand = useMemo(() => created ? `python cadia_web_mcp_bridge.py --server ${publicOrigin} --project ${projectId}` : '', [created, projectId, publicOrigin])
  const legacyConfig = useMemo(() => created ? JSON.stringify({
    mcpServers: {
      cadia: {
        command: 'python',
        args: ['C:/PATH/cadia_web_mcp_bridge.py', '--server', publicOrigin, '--project', projectId],
        env: { CADIA_MCP_TOKEN: created.token },
      },
    },
  }, null, 2) : '', [created, projectId, publicOrigin])

  return <div className="modal-backdrop" onMouseDown={(e) => e.currentTarget === e.target && onClose()}>
    <div className="modal-card mcp-modal">
      <button className="modal-close" onClick={onClose}><X size={18} /></button>
      <div className="connect-logo"><Network size={24} /></div>
      <div className="eyebrow">CADIA REMOTE MCP</div>
      <h2>Connect your own AI</h2>
      <p className="muted">Use your existing Claude, Cursor, Codex or another MCP client with this CADia project. Your external AI account pays its own AI usage; CADia exposes the same editable CAD tools and core used by the web app.</p>
      <div className="mcp-warning">Each token is locked to this project. Keep it private. CADia never needs your external AI password or API key.</div>

      {!created && <div className="mcp-create-row">
        <label>Connection name<input value={label} onChange={(e) => setLabel(e.target.value)} /></label>
        <label>Expires<select value={days} onChange={(e) => setDays(Number(e.target.value))}><option value={7}>7 days</option><option value={30}>30 days</option><option value={90}>90 days</option><option value={365}>1 year</option></select></label>
        <button className="primary" disabled={busy} onClick={create}><Plus size={16} />{busy ? 'Creating…' : 'Create connection'}</button>
      </div>}

      {created && <div className="mcp-created">
        <div className="mcp-once"><KeyRound size={17} /><span><strong>The token is shown only once.</strong><small>{created.warning}</small></span></div>

        <div className="mcp-step"><strong>Remote MCP URL</strong><div className="mcp-code-row"><code>{remoteUrl}</code><button onClick={() => copy('url', remoteUrl)}>{copied === 'url' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div></div>
        <div className="mcp-step"><strong>Bearer token</strong><div className="mcp-code-row"><code>{created.token}</code><button onClick={() => copy('token', created.token)}>{copied === 'token' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div></div>

        <div className="mcp-step"><strong>Claude / custom connector</strong><span className="muted">Add the Remote MCP URL as a custom connector, then add request header <code>Authorization</code> with the value below.</span><div className="mcp-code-row"><code>{authHeader}</code><button onClick={() => copy('auth', authHeader)}>{copied === 'auth' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div></div>

        <div className="mcp-step"><strong>Cursor / Windsurf / compatible MCP clients</strong><div className="mcp-config"><pre>{cursorConfig}</pre><button onClick={() => copy('cursor', cursorConfig)}>{copied === 'cursor' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div></div>

        <div className="mcp-step"><strong>Codex</strong><span className="muted">Set environment variable <code>CADIA_MCP_TOKEN</code> to the token above, then add this to your Codex config.</span><div className="mcp-config"><pre>{codexConfig}</pre><button onClick={() => copy('codex', codexConfig)}>{copied === 'codex' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div></div>

        <details className="mcp-step">
          <summary><strong>Legacy/local stdio bridge fallback</strong></summary>
          <p className="muted">Only use this when a client cannot connect to Remote MCP directly.</p>
          <button onClick={() => download('/api/mcp/bridge.py')}><Download size={15}/> cadia_web_mcp_bridge.py</button>
          <div className="mcp-code-row"><code>{legacyCommand}</code><button onClick={() => copy('legacy-cmd', legacyCommand)}>{copied === 'legacy-cmd' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div>
          <div className="mcp-config"><pre>{legacyConfig}</pre><button onClick={() => copy('legacy-config', legacyConfig)}>{copied === 'legacy-config' ? <Check size={15}/> : <Clipboard size={15}/>}</button></div>
        </details>
        <button className="secondary wide" onClick={() => setCreated(null)}>Create another token</button>
      </div>}

      <div className="mcp-token-list">
        <strong>Issued connections</strong>
        {tokens.length === 0 ? <span className="muted">No MCP tokens issued yet.</span> : tokens.map((row) => <div className={`mcp-token-item ${row.revokedAt ? 'revoked' : ''}`} key={row.id}>
          <span><b>{row.label}</b><small>{row.revokedAt ? 'Revoked' : `Expires ${row.expiresAt ? new Date(row.expiresAt).toLocaleDateString() : '-'}`}</small></span>
          {!row.revokedAt && <button title="Revoke token" onClick={() => revoke(row.id)}><Trash2 size={15}/></button>}
        </div>)}
      </div>
      {error && <div className="inline-error">{error}</div>}
    </div>
  </div>
}
