
#ifndef SGEMM_TUNER_H_INCLUDED
#define SGEMM_TUNER_H_INCLUDED

#include "config.h"
#include <vector>
#include <map>
#include <string>

using Configurations = std::pair<std::string, std::vector<size_t>>;
using Parameters = std::map<std::string, size_t>;

class OpenCL;

class Tuner {
    OpenCL & m_opencl;
    cl::Context m_context;
    cl::Device m_device;
public:
    std::string tune_sgemm(const int m, const int n, const int k,
                           const int batch_size, const int runs = 4);
    std::string load_sgemm_tuners(const int m, const int n, const int k,
                                  const int batch_size);

    static constexpr auto TUNER_VERSION = 0;
    Tuner(OpenCL & opencl, cl::Context context, cl::Device device) :
        m_opencl(opencl), m_context(context), m_device(device) {}
private:
    void store_sgemm_tuners(const int m, const int n, const int k,
                            const int batch_size, std::string tuners);
    bool valid_config_sgemm(Parameters p, bool exhaustive);
    std::string parameters_to_defines(const Parameters& p);
    std::string parameters_to_string(const Parameters& p);
    Parameters get_parameters_by_int(const std::vector<Configurations>& opts,
                                     const int n);
    std::string sgemm_tuners_from_line(std::string line, const int m,
                                       const int n, const int k,
                                       const int batch_size);
};

#endif
