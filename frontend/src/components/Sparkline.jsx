import { sparkPath } from '../lib/series'

// Inline trend line over a real derived series. `w`/`h` set the coordinate space;
// `full` stretches it to the container width. Renders an empty box when there aren't
// enough points to draw a line, rather than inventing a shape.
export default function Sparkline({ values, w, h, color, area = false, full = false, strokeWidth = 1.4, opacity = 0.85 }) {
  const width = full ? '100%' : w
  const d = sparkPath(values, w, h)
  if (!d) return <svg width={width} height={h} aria-hidden="true" style={{ flex: 'none' }} />

  return (
    <svg
      width={width}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      style={{ overflow: 'visible', flex: 'none', display: 'block' }}
    >
      {area && <path d={sparkPath(values, w, h, true)} fill={color} opacity="0.12" />}
      <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" opacity={opacity} />
    </svg>
  )
}
