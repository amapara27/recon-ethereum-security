import { useEffect, useRef, useState } from 'react'
import { getAlerts } from '../lib/api'

// Polls /api/get-alerts every `intervalMs` and reports connection status.
// `paused` freezes the feed in place (the topbar Live/Paused toggle) without dropping data.
export function useAlerts(intervalMs = 2000, paused = false) {
  const [alerts, setAlerts] = useState([])
  const [status, setStatus] = useState('connecting') // 'connecting' | 'online' | 'offline'
  const [updatedAt, setUpdatedAt] = useState(null)
  const timer = useRef(null)

  useEffect(() => {
    if (paused) return
    let active = true

    const tick = async () => {
      try {
        const data = await getAlerts()
        if (!active) return
        setAlerts(Array.isArray(data) ? data : [])
        setStatus('online')
        setUpdatedAt(Date.now())
      } catch {
        if (active) setStatus('offline')
      }
    }

    tick()
    timer.current = setInterval(tick, intervalMs)
    return () => {
      active = false
      clearInterval(timer.current)
    }
  }, [intervalMs, paused])

  return { alerts, status, updatedAt }
}
