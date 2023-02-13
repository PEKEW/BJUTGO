

#ifndef OPENCL_SCHEDULER_H_INCLUDED
#define OPENCL_SCHEDULER_H_INCLUDED
#include "config.h"

#include <vector>
#include <future>

#include "OpenCL.h"
#include "ThreadPool.h"

class OpenCLScheduler {
public:
    void initialize(const int channels);
    std::vector<std::unique_ptr<OpenCL_Network>> & get_networks() {
        return m_networks;
    }
    void forward(const std::vector<net_t>& input,
                 std::vector<net_t>& output_pol,
                 std::vector<net_t>& output_val);
private:
    class ForwardTask {
    public:
        const std::vector<net_t> *input;
        std::vector<net_t> * output;
        std::promise<void> prom;
        ForwardTask() : input(nullptr), output(nullptr) {}
        ForwardTask(const std::vector<net_t> * in,
                    std::vector<net_t> * out)
            : input(in), output(out) {}
    };

    std::vector<std::unique_ptr<OpenCL_Network>> m_networks;
    std::vector<std::unique_ptr<OpenCL>> m_opencl;
    Utils::ThreadPool m_threadpool;
};

extern OpenCLScheduler opencl;

#endif
