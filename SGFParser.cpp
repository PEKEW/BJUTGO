

#include "SGFParser.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

#include "SGFTree.h"
#include "Utils.h"

std::vector<std::string> SGFParser::chop_stream(std::istream& ins,
                                                size_t stopat) {
    std::vector<std::string> result;
    std::string gamebuff;

    ins >> std::noskipws;

    int nesting = 0;      // parentheses
    bool intag = false;   // brackets
    int line = 0;
    gamebuff.clear();

    char c;
    while (ins >> c && result.size() <= stopat) {
        if (c == '\n') line++;

        gamebuff.push_back(c);
        if (c == '\\') {
            // read literal char
            ins >> c;
            gamebuff.push_back(c);
            // Skip special char parsing
            continue;
        }

        if (c == '(' && !intag) {
            if (nesting == 0) {
                // eat ; too
                do {
                    ins >> c;
                } while (std::isspace(c) && c != ';');
                gamebuff.clear();
            }
            nesting++;
        } else if (c == ')' && !intag) {
            nesting--;

            if (nesting == 0) {
                result.push_back(gamebuff);
            }
        } else if (c == '[' && !intag) {
            intag = true;
        } else if (c == ']') {
            if (intag == false) {
                Utils::myprintf("Tag error on line %d", line);
            }
            intag = false;
        }
    }

    // No game found? Assume closing tag was missing (OGS)
    if (result.size() == 0) {
        result.push_back(gamebuff);
    }

    return result;
}

std::vector<std::string> SGFParser::chop_all(std::string filename,
                                             size_t stopat) {
    std::ifstream ins(filename.c_str(), std::ifstream::binary | std::ifstream::in);

    if (ins.fail()) {
        throw std::runtime_error("Error opening file");
    }

    auto result = chop_stream(ins, stopat);
    ins.close();

    return result;
}

// scan the file and extract the game with number index
std::string SGFParser::chop_from_file(std::string filename, size_t index) {
    auto vec = chop_all(filename, index);
    return vec[index];
}

std::string SGFParser::parse_property_name(std::istringstream & strm) {
    std::string result;

    char c;
    while (strm >> c) {
        // SGF property names are guaranteed to be uppercase,
        // except that some implementations like IGS are retarded
        // and don't folow the spec. So allow both upper/lowercase.
        if (!std::isupper(c) && !std::islower(c)) {
            strm.unget();
            break;
        } else {
            result.push_back(c);
        }
    }

    return result;
}

bool SGFParser::parse_property_value(std::istringstream & strm,
                                     std::string & result) {
    strm >> std::noskipws;

    char c;
    while (strm >> c) {
        if (!std::isspace(c)) {
            strm.unget();
            break;
        }
    }

    strm >> c;

    if (c != '[') {
        strm.unget();
        return false;
    }

    while (strm >> c) {
        if (c == ']') {
            break;
        } else if (c == '\\') {
            strm >> c;
        }
        result.push_back(c);
    }

    strm >> std::skipws;

    return true;
}

void SGFParser::parse(std::istringstream & strm, SGFTree * node) {
    bool splitpoint = false;

    char c;
    while (strm >> c) {
        if (strm.fail()) {
            return;
        }

        if (std::isspace(c)) {
            continue;
        }

        // parse a property
        if (std::isalpha(c) && std::isupper(c)) {
            strm.unget();

            std::string propname = parse_property_name(strm);
            bool success;

            do {
                std::string propval;
                success = parse_property_value(strm, propval);
                if (success) {
                    node->add_property(propname, propval);
                }
            } while (success);

            continue;
        }

        if (c == '(') {
            // eat first ;
            char cc;
            do {
                strm >> cc;
            } while (std::isspace(cc));
            if (cc != ';') {
                strm.unget();
            }
            // start a variation here
            splitpoint = true;
            // new node
            SGFTree * newptr = node->add_child();
            parse(strm, newptr);
        } else if (c == ')') {
            // variation ends, go back
            // if the variation didn't start here, then
            // push the "variation ends" mark back
            // and try again one level up the tree
            if (!splitpoint) {
                strm.unget();
                return;
            } else {
                splitpoint = false;
                continue;
            }
        } else if (c == ';') {
            // new node
            SGFTree * newptr = node->add_child();
            node = newptr;
            continue;
        }
    }
}
