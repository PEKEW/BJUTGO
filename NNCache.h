

#ifndef NNCACHE_H_INCLUDED
#define NNCACHE_H_INCLUDED

#include "config.h"

#include <deque>
#include <mutex>
#include <unordered_map>

#include "Network.h"

class NNCache {
public:
    // return the global NNCache
    static NNCache& get_NNCache(void);

    // Set a reasonable size gives max number of playouts
    void set_size_from_playouts(int max_playouts);

    // Resize NNCache
    void resize(int size);

    // Try and find an existing entry.
    bool lookup(std::uint64_t hash, Network::Netresult & result);

    // Insert a new entry.
    void insert(std::uint64_t hash,
                const Network::Netresult& result);

    // Return the hit rate ratio.
    std::pair<int, int> hit_rate() const {
        return {m_hits, m_lookups};
    }

    void dump_stats();

private:
    NNCache(int size = 150000);  // ~ 225MB

    std::mutex m_mutex;

    size_t m_size;

    // Statistics
    int m_hits{0};
    int m_lookups{0};
    int m_inserts{0};

    struct Entry {
        Entry(const Network::Netresult& r)
            : result(r) {}
        Network::Netresult result;  // ~ 1.5KB
    };

    // Map from hash to {features, result}
    std::unordered_map<std::uint64_t, std::unique_ptr<const Entry>> m_cache;
    // Order entries were added to the map.
    std::deque<size_t> m_order;
};

#endif
