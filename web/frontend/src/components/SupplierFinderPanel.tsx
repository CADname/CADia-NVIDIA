import { MapPin, X } from 'lucide-react'
import '../studio.css'
import '../supplier-finder.css'
import ManufacturingMarketplace from './ManufacturingMarketplace'

type Props = { projectId: string; modelReady?: boolean; onClose: () => void }

export default function SupplierFinderPanel({ projectId, modelReady, onClose }: Props) {
  return <div className="modal-backdrop cadia-studio-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section className="modal-card cadia-studio supplier-finder-modal" role="dialog" aria-modal="true" aria-label="Supplier finder">
      <header className="studio-header supplier-finder-header">
        <div className="studio-brand">
          <div className="studio-brand-icon"><MapPin size={20}/></div>
          <div><span className="studio-kicker">CADia / SUPPLIER FINDER</span><h2>Supplier Finder<span className="studio-header-dot">.</span></h2><p>Search 3D printing, CNC and prototyping services worldwide.</p></div>
        </div>
        <button className="studio-close" onClick={onClose} aria-label="Close supplier finder"><X size={19}/></button>
      </header>
      <div className="supplier-finder-scroll">
        <ManufacturingMarketplace projectId={projectId} modelReady={modelReady} mode="suppliers"/>
      </div>
    </section>
  </div>
}
