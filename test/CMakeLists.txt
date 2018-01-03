# CMake configuration for tests

# Sources
#file(GLOB cpps *.cpp)
#list(REMOVE_ITEM cpps "test_pch.cpp")
#file(GLOB headers *.h)
set(sources actual_evapotranspiration_test.cpp api_test.cpp calibration_test.cpp cell_builder_test.cpp dtss_test.cpp
    gamma_snow_test.cpp glacier_melt_test.cpp gridpp_test.cpp hbv_actual_evapotranspiration_test.cpp hbv_snow_test.cpp hbv_soil_test.cpp
    hbv_stack_test.cpp hbv_tank_test.cpp inverse_distance_test.cpp kalman_test.cpp kirchner_test.cpp kriging_test.cpp max_abs_average_accessor_test.cpp
    merge_test.cpp mocks.cpp predictor_test.cpp priestley_taylor_test.cpp pt_gs_k_test.cpp pt_hs_k_test.cpp pt_ss_k_test.cpp qm_test.cpp
    region_model_test.cpp routing_test.cpp runner.cpp sceua_test.cpp serialization_test.cpp skaugen_test.cpp time_axis_test.cpp 
    time_series_average_test.cpp time_series_fixup_test.cpp time_series_test.cpp utctime_utilities_test.cpp
)
set(target "test_shyft")

# Flags
add_definitions("-D__UNIT_TEST__ -DVERBOSE=0 ")
find_package(PNG REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(${target} ${sources})
# Additional libraries
target_link_libraries(${target} 
 shyftcore shyftapi 
 ${SHYFT_DEPENDENCIES}/lib/libboost_filesystem.so 
 ${SHYFT_DEPENDENCIES}/lib/libboost_system.so 
 ${SHYFT_DEPENDENCIES}/lib/libboost_serialization.so 
 ${SHYFT_DEPENDENCIES}/lib/libdlib.so 
 ${PNG_LIBRARY} ${LAPACK_LIBRARIES} 
)
#set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/../../shyft/lib")
#install(TARGETS ${target} DESTINATION ${CMAKE_SOURCE_DIR}/bin/Release)

# This can be made more specific, but we would need the list of tests.
#add_test(${target} ${target})
#The current list:
add_test(inverse_distance ${target} -nv --test-suite=inverse_distance)
add_test(time_axis ${target} -nv --test-suite=time_axis)
add_test(region_model ${target} -nv --test-suite=region_model)
add_test(priestley_taylor ${target} -nv --test-suite=priestley_taylor)
add_test(gamma_snow ${target} -nv --test-suite=gamma_snow)
add_test(kirchner ${target} -nv --test-suite=kirchner)
add_test(bayesian_kriging ${target} -nv --test-suite=bayesian_kriging)
add_test(utctime_utilities ${target} -nv --test-suite=utctime_utilities)
add_test(pt_gs_k ${target} -nv --test-suite=pt_gs_k)
add_test(actual_evapotranspiration ${target} -nv --test-suite=actual_evapotranspiration)
add_test(calibration ${target} -nv --test-suite=calibration)
add_test(hbv_snow  ${target} -nv --test-suite=hbv_snow)
add_test(pt_hs_k ${target} -nv --test-suite=pt_hs_k)
add_test(pt_ss_k ${target} -nv --test-suite=pt_ss_k)
add_test(time_series ${target} -nv --test-suite=time_series)
add_test(api ${target} -nv --test-suite=api)
add_test(cell_builder ${target} -nv --test-suite=cell_builder) 
add_test(skaugen ${target} -nv --test-suite=skaugen)
add_test(sceua ${target} -nv --test-suite=sceua)
add_test(gridpp ${target} -nv --test-suite=gridpp)
add_test(kalman ${target} -nv --test-suite=kalman)
add_test(hbv_tank ${target} -nv --test-suite=hbv_tank)
add_test(hbv_soil ${target} -nv --test-suite=hbv_soil)
add_test(hbv_actual_evapotranspiration ${target} -nv --test-suite=hbv_actual_evapotranspiration)
add_test(glacier_melt ${target} -nv --test-suite=glacier_melt)
add_test(kriging ${target} -nv --test-suite=kriging)
add_test(serialization ${target} -nv --test-suite=serialization)
add_test(routing ${target} -nv --test-suite=routing)
add_test(dtss ${target} -nv --test-suite=dtss)
add_test(qm ${target} -nv --test-suite=qm)
add_test(predictors ${target} -nv --test-suite=predictors)
add_test(ts_merge ${target} -nv --test-suite=ts_merge)


