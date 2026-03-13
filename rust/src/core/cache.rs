use lru::LruCache;
use std::num::NonZeroUsize;

/// LRU frame cache with memory-budget eviction.
///
/// Frames are keyed by frame index. When inserting a new frame would exceed
/// `max_memory_bytes`, the least-recently-used entries are evicted until there
/// is enough room.
pub struct FrameCache {
    cache: LruCache<usize, Vec<u8>>,
    max_memory_bytes: usize,
    current_memory_bytes: usize,
}

impl FrameCache {
    /// Create a new cache with the given memory budget in bytes.
    ///
    /// The internal LruCache capacity is set to `usize::MAX` so that the only
    /// eviction policy is the memory budget, not a fixed entry count.
    pub fn new(max_memory_bytes: usize) -> Self {
        // Use a large entry-count cap so only the memory budget drives eviction.
        let cap = NonZeroUsize::new(1_000_000).expect("non-zero");
        Self {
            cache: LruCache::new(cap),
            max_memory_bytes,
            current_memory_bytes: 0,
        }
    }

    /// Return a reference to the cached frame data for `frame_idx`, or `None`
    /// if the frame is not cached.  Accessing a frame promotes it to the most-
    /// recently-used position.
    pub fn get(&mut self, frame_idx: usize) -> Option<&Vec<u8>> {
        self.cache.get(&frame_idx)
    }

    /// Insert frame data into the cache.
    ///
    /// If the new frame would push total memory usage above the budget, the
    /// oldest (least-recently-used) entries are evicted until there is enough
    /// room.  If a frame with the same index already exists it is replaced and
    /// the memory accounting is updated accordingly.
    pub fn put(&mut self, frame_idx: usize, data: Vec<u8>) {
        // Remove existing entry so we can re-account its size.
        if let Some(old) = self.cache.pop(&frame_idx) {
            self.current_memory_bytes -= old.len();
        }

        let new_size = data.len();

        // Evict LRU entries until we have room for the new frame.
        while self.max_memory_bytes > 0
            && self.current_memory_bytes + new_size > self.max_memory_bytes
        {
            match self.cache.pop_lru() {
                Some((_, evicted)) => {
                    self.current_memory_bytes -= evicted.len();
                }
                None => break,
            }
        }

        self.current_memory_bytes += new_size;
        self.cache.put(frame_idx, data);
    }

    /// Remove all entries from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_memory_bytes = 0;
    }

    /// Return the number of frames currently in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Return `true` if the cache contains no frames.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Return the total number of bytes stored across all cached frames.
    pub fn memory_usage(&self) -> usize {
        self.current_memory_bytes
    }
}
