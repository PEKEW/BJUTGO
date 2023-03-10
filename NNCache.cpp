

#include "config.h"
#include <functional>

#include "NNCache.h"
#include "Utils.h"
#include "UCTSearch.h"

NNCache::NNCache(int size) : m_size(size) {}

NNCache& NNCache::get_NNCache(void) {
    static NNCache cache;
    return cache;
}

bool NNCache::lookup(std::uint64_t hash, Network::Netresult & result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_lookups;

    auto iter = m_cache.find(hash);
    if (iter == m_cache.end()) {
        return false;  // Not found.
    }

    const auto& entry = iter->second;

    // Found it.
    ++m_hits;
    result = entry->result;
    return true;
}

void NNCache::insert(std::uint64_t hash,
                     const Network::Netresult& result) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_cache.find(hash) != m_cache.end()) {
        return;  // Already in the cache.
    }

    m_cache.emplace(hash, std::make_unique<Entry>(result));
    m_order.push_back(hash);
    ++m_inserts;

    // If the cache is too large, remove the oldest entry.
    if (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::resize(int size) {
    m_size = size;
    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::set_size_from_playouts(int max_playouts) {
    // cache hits are generally from last several moves so setting cache
    // size based on playouts increases the hit rate while balancing memory
    // usage for low playout instances. 150'000 cache entries is ~225 MB
    constexpr auto num_cache_moves = 3;
    auto max_playouts_per_move =
        std::min(max_playouts,
                 UCTSearch::UNLIMITED_PLAYOUTS / num_cache_moves);
    auto max_size = num_cache_moves * max_playouts_per_move;
    max_size = std::min(150'000, std::max(6'000, max_size));
    NNCache::get_NNCache().resize(max_size);
}

void NNCache::dump_stats() {
    Utils::myprintf(
        "NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %u size\n",
        m_hits, m_lookups, 100. * m_hits / (m_lookups + 1),
        m_inserts, m_cache.size());
}
