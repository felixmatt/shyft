#pragma once
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <functional>
#include <cstring>
#include <regex>


#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>


#include <dlib/iosockstream.h>
#include <dlib/misc_api.h>


#include "api/time_series.h"
#include "time_series_info.h"
#include "utctime_utilities.h"

#include "dtss_url.h"
#include "dtss_msg.h"

namespace shyft {
namespace dtss {

using shyft::core::utctime;
using shyft::core::utcperiod;
using shyft::core::utctimespan;
using shyft::core::no_utctime;
using shyft::core::calendar;
using shyft::core::deltahours;

using gta_t = shyft::time_axis::generic_dt;
using gts_t = shyft::time_series::point_ts<gta_t>;

// TODO: Remove API dependency from core...
using shyft::api::apoint_ts;
using shyft::api::gpoint_ts;
using shyft::api::gts_t;
using shyft::api::aref_ts;

// ========================================

using ts_vector_t = shyft::api::ats_vector;
using ts_info_vector_t = std::vector<ts_info>;
using id_vector_t = std::vector<std::string>;

/** \brief a dtss client
 *
 * This class implements the client side functionality of the dtss client-server.
 *
 *
 */
struct client {
	dlib::iosockstream io;
	std::string host_port;
	explicit client(const std::string& host_port) :io(host_port), host_port(host_port) {}

	void reopen() {
		io.close();
		io.open(host_port);
	}
	void close(int timeout_ms = 1000) { io.close(timeout_ms); }

	std::vector<apoint_ts>
	percentiles(ts_vector_t const& tsv, utcperiod p, api::gta_t const&ta, const std::vector<int>& percentile_spec) {
		if (tsv.size() == 0)
			throw std::runtime_error("percentiles requires a source ts-vector with more than 0 time-series");
		if (percentile_spec.size() == 0)
			throw std::runtime_error("percentile function require more than 0 percentiles specified");
		if (!p.valid())
			throw std::runtime_error("percentiles require a valid period-specification");
		if (ta.size() == 0)
			throw std::runtime_error("percentile function require a time-axis with more than 0 steps");

		msg::write_type(message_type::EVALUATE_TS_VECTOR_PERCENTILES, io);
		{
			boost::archive::binary_oarchive oa(io);
			oa << p << tsv << ta << percentile_spec;
		}
		auto response_type = msg::read_type(io);
		if (response_type == message_type::SERVER_EXCEPTION) {
			auto re = msg::read_exception(io);
			throw re;
		} else if (response_type == message_type::EVALUATE_TS_VECTOR_PERCENTILES) {
			ts_vector_t r;
			{
				boost::archive::binary_iarchive ia(io);
				ia >> r;
			}
			return r;
		}
		throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
	}

	std::vector<apoint_ts>
	evaluate(ts_vector_t const& tsv, utcperiod p) {
		if (tsv.size() == 0)
			throw std::runtime_error("evaluate requires a source ts-vector with more than 0 time-series");
		if (!p.valid())
			throw std::runtime_error("percentiles require a valid period-specification");
		msg::write_type(message_type::EVALUATE_TS_VECTOR, io); {
			boost::archive::binary_oarchive oa(io);
			oa << p << tsv;
		}
		auto response_type = msg::read_type(io);
		if (response_type == message_type::SERVER_EXCEPTION) {
			auto re = msg::read_exception(io);
			throw re;
		} else if (response_type == message_type::EVALUATE_TS_VECTOR) {
			ts_vector_t r; {
				boost::archive::binary_iarchive ia(io);
				ia >> r;
			}
			return r;
		}
		throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
	}

	void store_ts(const ts_vector_t &tsv, bool overwrite_on_write, bool cache_on_write) {
		if (tsv.size() == 0)
			return; //trivial and considered valid case
					// verify that each member of tsv is a gpoint_ts
		for (auto const &ats : tsv) {
			auto rts = dynamic_cast<api::aref_ts*>(ats.ts.get());
			if (!rts) throw std::runtime_error(std::string("attempt to store a null ts"));
			if (rts->needs_bind()) throw std::runtime_error(std::string("attempt to store unbound ts:") + rts->id);
		}
		msg::write_type(message_type::STORE_TS, io);
		{
			boost::archive::binary_oarchive oa(io);
			oa << tsv << overwrite_on_write << cache_on_write;
		}
		auto response_type = msg::read_type(io);
		if (response_type == message_type::SERVER_EXCEPTION) {
			auto re = msg::read_exception(io);
			throw re;
		} else if (response_type == message_type::STORE_TS) {
			return;
		}
		throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
	}

	ts_info_vector_t find(const std::string& search_expression) {
		msg::write_type(message_type::FIND_TS, io);
		{
			//boost::archive::binary_oarchive oa(io);
			//oa << search_expression;
			msg::write_string(search_expression, io);
		}
		auto response_type = msg::read_type(io);
		if (response_type == message_type::SERVER_EXCEPTION) {
			auto re = msg::read_exception(io);
			throw re;
		} else if (response_type == message_type::FIND_TS) {
			ts_info_vector_t r;
			{
				boost::archive::binary_iarchive ia(io);
				ia >> r;
			}
			return r;
		}
		throw std::runtime_error(std::string("Got unexpected response:") + std::to_string((int)response_type));
	}

};

// ========================================

inline std::vector<apoint_ts> dtss_evaluate(const std::string& host_port, const ts_vector_t& tsv, utcperiod p, int timeout_ms = 10000) {
	return client(host_port).evaluate(tsv, p);
}

inline std::vector<apoint_ts> dtss_percentiles(const std::string& host_port, const ts_vector_t& tsv, utcperiod p, const api::gta_t& ta, std::vector<int> percentile_spec, int timeout_ms = 10000) {
	return client(host_port).percentiles(tsv, p, ta, percentile_spec);
}

} // shyft::dtss
} // shyft