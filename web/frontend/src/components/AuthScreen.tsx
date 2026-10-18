import { FormEvent, useState } from 'react'
import { ArrowRight, Box, Check, Cpu, LockKeyhole, Orbit } from 'lucide-react'
import { api } from '../api'
import type { User } from '../types'

type Props = { onAuthenticated: (user: User) => void }

export default function AuthScreen({ onAuthenticated }: Props) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const user = await api<User>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email: username, password }),
      })
      onAuthenticated(user)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-page global-auth">
      <section className="auth-visual">
        <div className="auth-orbit orbit-one" />
        <div className="auth-orbit orbit-two" />
        <div className="auth-copy">
          <div className="brand-lockup"><span className="brand-mark">C</span><span>CADia</span></div>
          <p className="eyebrow">AI-NATIVE PARAMETRIC CAD</p>
          <h1>Describe it.<br /><span>Engineer it.</span></h1>
          <p className="auth-lead">AI-native parametric CAD that turns engineering intent into editable B-Rep models, feature history, and assemblies.</p>
          <div className="feature-pills">
            <span><Cpu size={15} /> CadQuery / OCCT</span>
            <span><Orbit size={15} /> Face & Edge editing</span>
            <span><LockKeyhole size={15} /> Judge-only demo access</span>
          </div>
        </div>
        <div className="wire-object" aria-hidden="true">
          <Box size={220} strokeWidth={0.45} />
          <div className="wire-disc disc-a" />
          <div className="wire-disc disc-b" />
        </div>
      </section>
      <section className="auth-form-side">
        <form className="auth-card" onSubmit={submit}>
          <div className="mobile-brand"><span className="brand-mark">C</span> CADia</div>
          <p className="eyebrow">HACKATHON JUDGE ACCESS</p>
          <h2>Launch the CAD workspace</h2>
          <p className="muted">Enter the judge credentials provided with the hackathon submission. CADia AI is provided by the demo server, so no personal API key or ChatGPT connection is required.</p>
          <label>Judge ID<input autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Judge ID" required /></label>
          <label>Password<input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" required /></label>
          {error && <div className="form-error">{error}</div>}
          <button className="primary wide" disabled={busy}>
            {busy ? 'Signing in…' : 'Sign in to CADia'} <ArrowRight size={17} />
          </button>
          <div className="auth-note"><Check size={14} /> Guest access and public account registration are disabled for this judge build.</div>
        </form>
      </section>
    </main>
  )
}
