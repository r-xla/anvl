#include <testthat.h>

#include <functional>
#include <string>

#include "lru_cache.h"

using IntCache = anvl::LRUCache<int, int, std::hash<int>, std::equal_to<int>>;
using StrCache = anvl::LRUCache<std::string, int, std::hash<std::string>,
                                 std::equal_to<std::string>>;

context("LRUCache") {
  test_that("a fresh cache is empty") {
    IntCache c(10);
    expect_true(c.size() == 0u);
  }

  test_that("set then get returns the value; a miss returns nullptr") {
    StrCache c(2);
    c.set("a", 1);
    c.set("b", 2);
    expect_true(*c.get("a") == 1);
    expect_true(*c.get("b") == 2);
    expect_true(c.get("c") == nullptr);  // miss (xlamisc: default / NULL)
  }

  test_that("get() acts as membership: non-null for present, null for absent") {
    StrCache c(1);
    c.set("a", 1);
    expect_true(c.get("a") != nullptr);  // has("a")
    expect_true(c.get("b") == nullptr);  // !has("b")
  }

  test_that("clear empties the cache") {
    StrCache c(2);
    c.set("a", 1);
    c.set("b", 2);
    expect_true(c.size() == 2u);
    c.clear();
    expect_true(c.size() == 0u);
    expect_true(c.get("a") == nullptr);
    expect_true(c.get("b") == nullptr);
  }

  test_that("set on an existing key updates the value without growing") {
    StrCache c(2);
    c.set("a", 1);
    c.set("a", 11);
    expect_true(*c.get("a") == 11);
    expect_true(c.size() == 1u);
  }

  // xlamisc's "LRU order is maintained (MRU -> LRU)" sequence, verified by the
  // eviction victim (we cannot read keys_mru_to_lru() here).
  test_that("MRU->LRU order is maintained, so the LRU entry is evicted") {
    StrCache c(3);
    c.set("a", 1);
    expect_true(*c.get("a") == 1);
    c.set("b", 2);
    expect_true(*c.get("b") == 2);
    c.set("c", 3);
    expect_true(*c.get("c") == 3);  // order: c, b, a
    expect_true(*c.get("a") == 1);  // touch a -> order: a, c, b
    c.set("b", 22);
    expect_true(*c.get("b") == 22);  // update b -> order: b, a, c
    c.set("c", 33);                  // update c -> order: c, b, a
    c.set("d", 4);                   // overflow (cap 3) -> evict LRU "a"
    expect_true(*c.get("d") == 4);
    expect_true(c.get("a") == nullptr);  // "a" was LRU: evicted
    expect_true(*c.get("b") == 22);
    expect_true(*c.get("c") == 33);
    expect_true(*c.get("d") == 4);
    expect_true(c.size() == 3u);
  }

  test_that("on_evict fires on overwrite, capacity eviction, and clear") {
    int evicted = 0;
    IntCache c(2, [&](int&) { ++evicted; });
    c.set(1, 10);
    c.set(2, 20);
    c.set(1, 11);  // overwrite -> on_evict on the old value
    expect_true(evicted == 1);
    c.set(3, 30);  // over capacity -> evict LRU (2)
    expect_true(evicted == 2);
    c.clear();  // evict the two survivors (1, 3)
    expect_true(evicted == 4);
    expect_true(c.size() == 0u);
  }
}
