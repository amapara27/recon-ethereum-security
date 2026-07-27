// Self-check for the derived-series helpers. Run: node src/lib/series.test.js
import assert from 'node:assert/strict'
import { sparkPath, timeBuckets, meanProb, peakProb, indexByAddress, seriesFor, counterparties } from './series.js'

const at = (minsAgo) => new Date(Date.now() - minsAgo * 60_000).toISOString()

// Newest first, the order /api/get-alerts returns.
const feed = [
  { tx_hash: 'c', address: '0xAA', to_address: '0xCC', value: '2', probability: 0.9, timestamp: at(1) },
  { tx_hash: 'b', address: '0xBB', to_address: '0xaa', value: '1', probability: 0.5, timestamp: at(30) },
  { tx_hash: 'a', address: '0xaa', to_address: '0xBB', value: '3', probability: 0.1, timestamp: at(90) },
]

// sparkPath refuses to invent a shape it doesn't have the points for
assert.equal(sparkPath([], 10, 10), '')
assert.equal(sparkPath([0.5], 10, 10), '', 'one point is not a line')
assert.match(sparkPath([0, 1], 10, 10), /^M0\.0 \d/)
assert.ok(sparkPath([0, 1], 10, 10, true).endsWith('L10 10 L0 10 Z'), 'closed path returns an area')
assert.doesNotMatch(sparkPath([0, 0, 0], 10, 10), /NaN/, 'an all-zero series must not divide by zero')

// timeBuckets: oldest bucket first, out-of-window rows dropped, empty buckets kept
const counts = timeBuckets(feed, 2, 60 * 60 * 1000, (r) => r.length)
assert.deepEqual(counts, [1, 1], 'the 90-minute-old row falls outside a 1h window')
assert.equal(timeBuckets(feed, 4, 60 * 60 * 1000, (r) => r.length).length, 4, 'series keeps a fixed length')
assert.equal(meanProb([]), 0)
assert.equal(peakProb([]), 0)
assert.equal(peakProb(feed), 0.9)

// indexByAddress: case-insensitive, oldest-first, one entry per row per address
const index = indexByAddress(feed)
assert.deepEqual(seriesFor(index, '0xaa'), [0.1, 0.5, 0.9], 'oldest first, both directions counted')
assert.deepEqual(seriesFor(index, '0xAA'), seriesFor(index, '0xaa'), 'lookup is case-insensitive')
assert.deepEqual(seriesFor(index, '0xdead'), [], 'unknown address has no series')
assert.deepEqual(
  indexByAddress([{ address: '0xaa', to_address: '0xAA', probability: 0.4 }]).get('0xaa'),
  [0.4],
  'a self-transfer counts once',
)

// counterparties: de-duplicated, newest first, never the address itself
assert.deepEqual(counterparties(feed, '0xaa').map((p) => p.address), ['0xCC', '0xBB'])
assert.deepEqual(counterparties(feed, '0xaa', 1).map((p) => p.address), ['0xCC'], 'limit is respected')
assert.deepEqual(counterparties(feed, ''), [])

console.log('series.js: all checks passed')
