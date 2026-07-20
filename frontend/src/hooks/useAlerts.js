import { useEffect, useRef, useState } from 'react'
import { getAlerts } from '../lib/api'

// Polls /api/get-alerts every `intervalMs` and reports connection status.
export function useAlerts(intervalMs = 2000) {
  const [alerts, setAlerts] = useState([])
  const [status, setStatus] = useState('connecting') // 'connecting' | 'online' | 'offline'
  const timer = useRef(null)

  useEffect(() => {
    let active = true

    const tick = async () => {
      try {
        const data = await getAlerts()
        if (!active) return
        setAlerts(Array.isArray(data) ? data : [])
        setStatus('online')
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
  }, [intervalMs])

  return { alerts, status }
}
