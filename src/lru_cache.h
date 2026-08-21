// A generic least-recently-used cache: a hashmap for lookup plus a
// doubly-linked list ordering entries most- to least-recently-used (mirrors
// xlamisc::LRUCache).

#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <unordered_map>
#include <utility>

namespace anvl {

// `get`/`set` move the touched entry to the front; `set` evicts from the back
// when over capacity. `on_evict` runs on the evicted value before it is dropped
// -- used to release R objects held in the value.
template <typename K, typename V, typename Hash, typename Eq>
class LRUCache {
 public:
  explicit LRUCache(std::size_t capacity,
                    std::function<void(V&)> on_evict = nullptr)
      // move avoids copying the std::function's heap-allocated target
      : capacity_(capacity), on_evict_(std::move(on_evict)) {}

  // Returns a pointer to the value (now MRU), or nullptr on miss.
  V* get(const K& key) {
    // the hashmap esentially stores pointers into the linked list
    auto it = index_.find(key);
    if (it == index_.end()) return nullptr;
    // move this entry's node to the front, marking it most-recently-used
    // I.e., we move it-> second before order_.begin() within order_
    // We call it->second because it
    order_.splice(order_.begin(), order_, it->second);
    return &it->second->value;
  }

  void set(const K& key, V value) {
    auto it = index_.find(key);  // returns an iterator to a (key, entry) pair
    // so both it->first is the key and it->second->key is the key.
    // it->second->value is the value
    if (it !=
        index_.end()) {  // key already present: update value and move to front
      if (on_evict_) on_evict_(it->second->value);
      it->second->value = std::move(value);
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    // not already present: add to the front
    order_.push_front(Entry{key, std::move(value)});
    // and register it in the hashmap
    index_.emplace(key, order_.begin());
    if (index_.size() > capacity_) {  // remove LRU
      Entry& victim = order_.back();
      // Erase from the index BEFORE on_evict: erasing re-hashes and compares
      // the key, and on_evict may release resources the key's hash/equality
      // read.
      index_.erase(victim.key);
      if (on_evict_) on_evict_(victim.value);
      order_.pop_back();
    }
  }

  std::size_t size() const { return index_.size(); }

  // Run on_evict over every entry and drop them.
  void clear() {
    if (on_evict_) {
      for (Entry& e : order_) on_evict_(e.value);
    }
    order_.clear();
    index_.clear();
  }

 private:
  struct Entry {
    K key;
    V value;
  };
  // doubly linked list: it's cheap to append to the front (MRU) and pop from
  // the back (LRU)
  std::list<Entry> order_;
  // key -> position in `order_`, for O(1) lookup. The four type params are:
  // key type (K), mapped value (a `order_` iterator), hash functor (Hash),
  // and key-equality functor (Eq).
  //
  // The mapped value is a `list<Entry>::iterator`: a handle pointing at one
  // specific node inside `order_` (dereference it to reach that Entry). We
  // store this handle -- rather than the value itself or an index -- so a
  // lookup lands directly on the entry's list node: get()/set() can then
  // splice it to the front in O(1) without copying the value.
  std::unordered_map<K, typename std::list<Entry>::iterator, Hash, Eq> index_;
  std::size_t capacity_;
  std::function<void(V&)> on_evict_;
};

}  // namespace anvl
