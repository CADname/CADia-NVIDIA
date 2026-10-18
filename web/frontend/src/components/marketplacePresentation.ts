// Human-readable presentation of provider MCP responses. No fabricated prices or IDs.
export type MaterialOption = { id: string; material: string; color: string; hex: string }

function parsedJson(text: string): unknown {
  const stripped = text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')
  if (!stripped || !/^[\[{]/.test(stripped)) return undefined
  try { return JSON.parse(stripped) as unknown } catch { return undefined }
}
function objectOf(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
function field(value: unknown, max = 160): string {
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (typeof value !== 'string') return ''
  return value.replace(/[\x00-\x1f\x7f]/g, ' ').trim().slice(0, max)
}
export function materialOptions(raw: unknown): MaterialOption[] {
  const result: MaterialOption[] = []; const seen = new Set<string>()
  let examined = 0
  function visit(value: unknown, depth: number) {
    if (depth > 12 || examined++ > 5000 || result.length >= 1200) return
    if (typeof value === 'string') { const parsed=parsedJson(value); if(parsed!==undefined) visit(parsed, depth+1); return }
    if (Array.isArray(value)) { for(const item of value) visit(item,depth+1); return }
    const entry=objectOf(value); if(!entry) return
    const id=field(entry.id || entry.filament_id || entry.filamentId, 120)
    const material=field(entry.material || entry.material_name || entry.materialName || entry.type, 65)
    const color=field(entry.color || entry.colour || entry.color_name || entry.colorName,65)
    const hex=field(entry.hex || entry.hex_color || entry.hexColor,12)
    if(id && material && !seen.has(id)) {
      seen.add(id); result.push({id,material,color:color || 'Unspecified',hex:/^#[a-fA-F0-9]{6}$/.test(hex)?hex:''})
    }
    for(const [key,child] of Object.entries(entry)) {
      if (['id','filament_id','filamentId','hex','hex_color','hexColor','material','material_name','materialName','type','color','colour','color_name','colorName'].includes(key)) continue
      if (child && typeof child==='object' || key==='text' && typeof child==='string') visit(child,depth+1)
    }
  }
  visit(raw,0)
  return result.sort((a,b)=>a.material.localeCompare(b.material)||a.color.localeCompare(b.color))
}

const supportedFields: Array<[string,string[]]> = [
 ['Quote reference', ['quote_id','quoteId','quoteID']],
 ['Status', ['status','quote_status']],
 ['Material', ['material']],
 ['Color', ['color','colour']],
 ['Quantity', ['quantity']],
 ['Currency', ['currency','currency_code']],
 ['Total price', ['total_price','total_cost','totalPrice','price_total','total']],
 ['Unit price', ['unit_price','unit_cost','unitPrice']],
 ['Subtotal', ['subtotal']],
 ['Shipping cost', ['shipping_cost','shipping_price','shipping_fee','shippingCost']],
 ['Lead time', ['lead_time','lead_time_days','leadTime']],
 ['Estimated delivery', ['estimated_delivery','delivery_estimate','estimatedDelivery']],
]
export function quoteFields(raw:unknown):Array<{label:string,value:string}> {
  const found=new Map<string,string>(); let examined=0
  function walk(value: unknown, depth:number){
    if(depth>12 || examined++>3000)return
    if(typeof value==='string'){const parsed=parsedJson(value); if(parsed!==undefined)walk(parsed,depth+1);return}
    if(Array.isArray(value)){for(const item of value)walk(item,depth+1);return}
    const obj=objectOf(value);if(!obj)return
    for(const [label,keys] of supportedFields){
      if(found.has(label))continue
      for(const key of keys) {const v=field(obj[key],200); if(v){found.set(label,v);break}}
    }
    for(const [key,child] of Object.entries(obj)){
      if(['stl_base64','address','email','name','line1','line2','api_key','token'].includes(key))continue
      if(child && typeof child==='object' || key==='text' && typeof child==='string')walk(child,depth+1)
    }
  }
  walk(raw,0)
  return supportedFields.filter(([label])=>found.has(label)).map(([label])=>({label,value:found.get(label)!}))
}
