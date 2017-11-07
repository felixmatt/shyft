#pragma once
#include <string>
#include <cstdint>
#include <exception>

namespace shyft {
namespace dtss {

/** \brief dtss message-types 
 *
 * The message types used for the wire-communication of dtss.
 * 
 */
enum class message_type : uint8_t {
	SERVER_EXCEPTION,
	EVALUATE_TS_VECTOR,
	EVALUATE_TS_VECTOR_PERCENTILES,
	FIND_TS,
	STORE_TS,
	// EVALUATE_TS_VECTOR_HISTOGRAM //-- tsv,period,ta,bin_min,bin_max -> ts_vector[n_bins]
};

// ========================================

namespace msg {

/** stream utility functions for reading basic message-types/parts
 * 
 */

template <class T>
message_type read_type(T& in) {
	int32_t mtype;
	in.read((char*)&mtype, sizeof(mtype));
	return (message_type)mtype;
}

template <class T>
void write_type(message_type mt, T& out) {
	int32_t mtype = (int32_t)mt;
	out.write((const char *)&mtype, sizeof(mtype));
}

template <class T>
void write_string(const std::string& s, T& out) {
	int32_t sz = s.size();
	out.write((const char*)&sz, sizeof(sz));
	out.write(s.data(), sz);
}

template <class T>
std::string read_string(T& in) {
	std::int32_t sz;
	in.read((char*)&sz, sizeof(sz));
	std::string msg(sz, '\0');
	in.read((char*)msg.data(), sz);
	return msg;
}

template <class T>
void write_exception(const std::exception& e, T& out) {
	int32_t sz = strlen(e.what());
	out.write((const char*)&sz, sizeof(sz));
	out.write(e.what(), sz);
}

template <class T>
void send_exception(const std::exception& e, T& out) {
	write_type(message_type::SERVER_EXCEPTION, out);
	int32_t sz = strlen(e.what());
	out.write((const char*)&sz, sizeof(sz));
	out.write(e.what(), sz);
}

template <class T>
std::runtime_error read_exception(T& in) {
	int32_t sz;
	in.read((char*)&sz, sizeof(sz));
	std::string msg(sz, '\0');
	in.read((char*)msg.data(), sz);
	return std::runtime_error(msg);
}

}  // msg
}
}