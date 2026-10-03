import { Camera, Ruler, ScanLine, X } from 'lucide-react'

type Props = {
  onClose: () => void
  onDrawing: () => void
  onPhoto: () => void
}

export default function OpenCVVisionLauncher({ onClose, onDrawing, onPhoto }: Props) {
  return (
    <div className="modal-backdrop visual-cad-backdrop" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose() }}>
      <section className="visual-cad-modal opencv-launcher-modal">
        <header>
          <div>
            <span className="visual-cad-icon opencv-launcher-icon"><ScanLine size={20} /></span>
            <div><h2>OpenCV 5 Vision CAD</h2><p>Agentic perception → native editable B-Rep → deterministic geometric verification</p></div>
          </div>
          <button className="icon-button" onClick={onClose}><X size={18} /></button>
        </header>

        <div className="opencv-launcher-status">
          <span><i /> OpenCV 5 lane</span>
          <small>Existing Standard CADia and NVIDIA/Cosmos/Nemotron paths remain unchanged.</small>
        </div>

        <div className="opencv-launcher-grid">
          <button onClick={onDrawing}>
            <span className="opencv-launcher-card-icon"><Ruler size={23} /></span>
            <strong>Drawing → CAD</strong>
            <small>Deskew · line/circle/arc evidence · dimension candidates · render verification · bounded repair</small>
            <em>OPEN WITH OPENCV 5</em>
          </button>
          <button onClick={onPhoto}>
            <span className="opencv-launcher-card-icon"><Camera size={23} /></span>
            <strong>Photo → CAD</strong>
            <small>Multi-view contour/circle evidence · candidate ranking · source↔render verification · repair gate</small>
            <em>OPEN WITH OPENCV 5</em>
          </button>
        </div>

        <div className="opencv-launcher-flow">
          <span>DETECT</span><b>→</b><span>BUILD</span><b>→</b><span>MEASURE</span><b>→</b><span>REPAIR</span><b>→</b><span>VERIFY</span>
        </div>
      </section>
    </div>
  )
}
