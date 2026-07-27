// Time-series derived from the live alert feed. Everything here is computed from real
// /api/get-alerts rows — the design mock generated these numbers, we bucket the real ones.
import { toMs } from './format.js' // explicit extension so series.test.js runs under plain node

// SVG path through `vals` (oldest → newest) scaled to a w×h box. `close` returns a filled area.
// Fewer than two points has no line to draw, so callers get an empty path and render nothing.
export function sparkPath(vals, w, h, close = false) {
  if (!vals || vals.length < 2) return ''
  const max = Math.max(...vals, 1e-3)
  const pts = vals.map((v, i) => [(i / (vals.length - 1)) * w, h - (v / max) * (h - 2) - 1])
  let d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ')
  if (close) d += ` L${w} ${h} L0 ${h} Z`
  return d
}

// Split alerts into `count` equal time buckets ending now, oldest first, and reduce each.
// Alerts older than the window are dropped; empty buckets still get a reduce() call so the
// series keeps a fixed length and a gap reads as a gap.
export function timeBuckets(alerts, count, spanMs, reduce) {
  const now = Date.now()
  const width = spanMs / count
  const bins = Array.from({ length: count }, () => [])
  for (const a of alerts) {
    const t = toMs(a.timestamp)
    if (Number.isNaN(t)) continue
    const i = count - 1 - Math.floor((now - t) / width)
    if (i >= 0 && i < count) bins[i].push(a)
  }
  return bins.map(reduce)
}

export const meanProb = (rows) => (rows.length ? rows.reduce((s, a) => s + (a.probability || 0), 0) / rows.length : 0)
export const peakProb = (rows) => (rows.length ? Math.max(...rows.map((a) => a.probability || 0)) : 0)

// address (lowercased) → its scores over the window, oldest first. Built once per feed
// so the table can draw a per-address trend on every row without rescanning the feed.
export function indexByAddress(alerts) {
  const map = new Map()
  for (let i = alerts.length - 1; i >= 0; i--) {
    // the API returns newest first, so walking backwards yields oldest-first series
    const r = alerts[i]
    const p = r.probability || 0
    const seen = new Set()
    for (const a of [r.address, r.to_address]) {
      if (!a) continue
      const k = a.toLowerCase()
      if (seen.has(k)) continue
      seen.add(k)
      const arr = map.get(k)
      if (arr) arr.push(p)
      else map.set(k, [p])
    }
  }
  return map
}

export const seriesFor = (index, address) => (address ? index.get(address.toLowerCase()) || [] : [])

// Every alert involving `address` (as sender or recipient), oldest first.
export function addressTouches(alerts, address) {
  if (!address) return []
  const a = address.toLowerCase()
  return alerts
    .filter((r) => r.address?.toLowerCase() === a || r.to_address?.toLowerCase() === a)
    .slice()
    .sort((x, y) => toMs(x.timestamp) - toMs(y.timestamp))
}

// Counterparties seen with `address` in the window, most recent first, de-duplicated.
export function counterparties(alerts, address, limit = 4) {
  const a = address?.toLowerCase()
  if (!a) return []
  const seen = new Map()
  for (const r of alerts) {
    const from = r.address?.toLowerCase()
    const to = r.to_address?.toLowerCase()
    if (from !== a && to !== a) continue
    const other = from === a ? r.to_address : r.address
    if (!other || other.toLowerCase() === a || seen.has(other.toLowerCase())) continue
    seen.set(other.toLowerCase(), { address: other, probability: r.probability, value: r.value })
    if (seen.size >= limit) break
  }
  return [...seen.values()]
}
