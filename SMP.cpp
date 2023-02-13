

#include "SMP.h"

#include <cassert>
#include <thread>

SMP::Mutex::Mutex() {
    m_lock = false;
}

SMP::Lock::Lock(Mutex & m) {
    m_mutex = &m;
    lock();
}

void SMP::Lock::lock() {
    assert(!m_owns_lock);
    while (m_mutex->m_lock.exchange(true, std::memory_order_acquire) == true);
    m_owns_lock = true;
}

void SMP::Lock::unlock() {
    assert(m_owns_lock);
    auto lock_held = m_mutex->m_lock.exchange(false, std::memory_order_release);

    // If this fails it means we are unlocking an unlocked lock
#ifdef NDEBUG
    (void)lock_held;
#else
    assert(lock_held);
#endif
    m_owns_lock = false;
}

SMP::Lock::~Lock() {
    // If we don't claim to hold the lock,
    // don't bother trying to unlock in the destructor.
    if (m_owns_lock) {
        unlock();
    }
}

int SMP::get_num_cpus() {
    return std::thread::hardware_concurrency();
}
