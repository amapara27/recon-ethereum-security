# Minimal check for the per-block address cap. Run: python test_pick_addresses.py
import types

import monitor

def block(senders):
    return types.SimpleNamespace(transactions=[{'from': s} for s in senders])

def test_dedupes_and_caps():
    monitor.processed_addr.clear()
    monitor.MAX_ADDR_PER_BLOCK = 3

    picked = monitor.pick_addresses(block(['a', 'a', 'b', 'c', 'd', 'e']))
    assert list(picked) == ['a', 'b', 'c'], picked   # deduped, capped at 3

    monitor.processed_addr.update({'a', 'b'})        # already-scored senders are skipped
    picked = monitor.pick_addresses(block(['a', 'b', 'c', 'd']))
    assert list(picked) == ['c', 'd'], picked

    print("ok")

if __name__ == "__main__":
    test_dedupes_and_caps()
