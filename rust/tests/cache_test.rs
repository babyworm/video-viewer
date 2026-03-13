use video_viewer::core::cache::FrameCache;

#[test]
fn test_cache_put_get() {
    let mut cache = FrameCache::new(1024 * 1024);
    let data = vec![1u8, 2, 3, 4, 5];
    cache.put(0, data.clone());
    assert_eq!(cache.get(0), Some(&data));
}

#[test]
fn test_cache_miss() {
    let mut cache = FrameCache::new(1024 * 1024);
    assert_eq!(cache.get(42), None);
}

#[test]
fn test_cache_eviction() {
    // Budget: 300 bytes, frames: 100 bytes each.
    // Insert 5 frames → oldest 2 must be evicted to stay within budget.
    let mut cache = FrameCache::new(300);

    for i in 0..5usize {
        cache.put(i, vec![0u8; 100]);
    }

    // Only 3 most-recent frames should survive.
    assert_eq!(cache.len(), 3);
    assert!(cache.memory_usage() <= 300);

    // Frames 0 and 1 were the oldest → evicted.
    assert_eq!(cache.get(0), None);
    assert_eq!(cache.get(1), None);

    // Frames 2, 3, 4 should still be present.
    assert!(cache.get(2).is_some());
    assert!(cache.get(3).is_some());
    assert!(cache.get(4).is_some());
}

#[test]
fn test_cache_clear() {
    let mut cache = FrameCache::new(1024 * 1024);
    cache.put(0, vec![1u8; 100]);
    cache.put(1, vec![2u8; 200]);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.get(0), None);
    assert_eq!(cache.get(1), None);
}

#[test]
fn test_cache_memory_usage() {
    let mut cache = FrameCache::new(1024 * 1024);
    cache.put(0, vec![0u8; 1000]);
    cache.put(1, vec![0u8; 2000]);
    assert_eq!(cache.memory_usage(), 3000);
}

#[test]
fn test_cache_lru_order() {
    // Budget: 300 bytes, frames: 100 bytes each.
    // Insert frames 0, 1, 2 (fills budget).
    // Access frame 0 to make it recently used.
    // Insert frame 3 → frame 1 (now LRU) should be evicted, not frame 0.
    let mut cache = FrameCache::new(300);

    cache.put(0, vec![0u8; 100]);
    cache.put(1, vec![1u8; 100]);
    cache.put(2, vec![2u8; 100]);

    // Promote frame 0 to most-recently-used.
    assert!(cache.get(0).is_some());

    // Insert frame 3 — needs to evict the LRU entry, which is now frame 1.
    cache.put(3, vec![3u8; 100]);

    assert_eq!(cache.len(), 3);
    assert_eq!(cache.get(1), None, "frame 1 should have been evicted (LRU)");
    assert!(cache.get(0).is_some(), "frame 0 should still be cached");
    assert!(cache.get(2).is_some(), "frame 2 should still be cached");
    assert!(cache.get(3).is_some(), "frame 3 should be cached");
}
