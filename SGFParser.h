

#ifndef SGFPARSER_H_INCLUDED
#define SGFPARSER_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

#include "SGFTree.h"

class SGFParser {
private:
    static std::string parse_property_name(std::istringstream & strm);
    static bool parse_property_value(std::istringstream & strm, std::string & result);
public:
    static std::string chop_from_file(std::string fname, size_t index);
    static std::vector<std::string> chop_all(std::string fname,
                                             size_t stopat = SIZE_MAX);
    static std::vector<std::string> chop_stream(std::istream& ins,
                                                size_t stopat = SIZE_MAX);
    static void parse(std::istringstream & strm, SGFTree * node);
};


#endif
