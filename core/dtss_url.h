#pragma once
#include <string>
#include <algorithm>

namespace shyft {
namespace dtss {

// TODO: inline when vs implements P0386R2: Inline variables
extern std::string shyft_prefix;//="shyft://";  ///< marks all internal handled urls


/**construct a shyft-url from container and ts-name */
inline std::string shyft_url(const std::string& container, const std::string& ts_name) {
	return shyft_prefix + container + "/" + ts_name;
}

/** match & extract fast the following 'shyft://<container>/'
* \param url like pattern above
* \return <container> or empty string if no match
*/
inline std::string extract_shyft_url_container(const std::string& url) {
	if ((url.size() < shyft_prefix.size() + 2) || !std::equal(begin(shyft_prefix), end(shyft_prefix), begin(url)))
		return std::string{};
	auto ce = url.find_first_of('/', shyft_prefix.size());
	if (ce == std::string::npos)
		return std::string{};
	return url.substr(shyft_prefix.size(), ce - shyft_prefix.size());
}

}
}