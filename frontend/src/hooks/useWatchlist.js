import { useCallback, useEffect, useState } from 'react'

const KEY = 'recon-watchlist'
export const ADDRESS_RE = /^0x[0-9a-fA-F]{40}$/

// Pinned addresses, persisted in localStorage. There is no watchlist endpoint yet —
// scores and drift are read off the live alert feed, so nothing here is stored server-side.
// ponytail: localStorage, single browser. Move to the backend when accounts exist.
function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY))
    return Array.isArray(raw) ? raw.filter((a) => ADDRESS_RE.test(a)).map((a) => a.toLowerCase()) : []
  } catch {
    return []
  }
}

export function useWatchlist() {
  const [list, setList] = useState(load)

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(list))
    } catch {
      /* ignore private-mode storage errors */
    }
  }, [list])

  const add = useCallback((address) => {
    const a = address?.trim().toLowerCase()
    if (!ADDRESS_RE.test(a || '')) return false
    setList((l) => (l.includes(a) ? l : [a, ...l]))
    return true
  }, [])

  const remove = useCallback((address) => {
    const a = address?.toLowerCase()
    setList((l) => l.filter((x) => x !== a))
  }, [])

  return { list, add, remove }
}
