cmake_minimum_required(VERSION 3.21)
project(askme_webrtc_apm LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(ASKME_WEBRTC_APM_COMMIT "846fe90a289f58b7c9303a635142aa2c7caa93e5")
set(WEBRTC_APM_ROOT "$ENV{WEBRTC_APM_ROOT}" CACHE PATH
    "Prefix created by scripts/build_dependency for the pinned WebRTC APM source")

if(NOT WEBRTC_APM_ROOT)
  message(FATAL_ERROR
    "WEBRTC_APM_ROOT is required. Build the pinned dependency first; see docs/AEC_BUILD.md")
endif()

set(_provenance "${WEBRTC_APM_ROOT}/share/askme-webrtc-apm/SOURCE_COMMIT.txt")
if(NOT EXISTS "${_provenance}")
  message(FATAL_ERROR
    "Missing ${_provenance}; refusing to build against an unverified WebRTC APM library")
endif()
file(READ "${_provenance}" _actual_commit)
string(STRIP "${_actual_commit}" _actual_commit)
if(NOT _actual_commit STREQUAL ASKME_WEBRTC_APM_COMMIT)
  message(FATAL_ERROR
    "WebRTC APM provenance mismatch: expected ${ASKME_WEBRTC_APM_COMMIT}, got ${_actual_commit}")
endif()

find_path(WEBRTC_APM_INCLUDE_DIR
  NAMES api/audio/audio_processing.h
  PATHS "${WEBRTC_APM_ROOT}/include/webrtc-audio-processing-2"
  NO_DEFAULT_PATH
  REQUIRED)
find_library(WEBRTC_APM_LIBRARY
  NAMES webrtc-audio-processing-2
  PATHS "${WEBRTC_APM_ROOT}/lib" "${WEBRTC_APM_ROOT}/lib64"
  NO_DEFAULT_PATH
  REQUIRED)

pybind11_add_module(_askme_webrtc_apm MODULE src/bindings.cpp)
target_compile_features(_askme_webrtc_apm PRIVATE cxx_std_17)
target_include_directories(_askme_webrtc_apm PRIVATE
  "${WEBRTC_APM_INCLUDE_DIR}"
  # Public WebRTC APM headers include Abseil headers installed at this root.
  "${WEBRTC_APM_ROOT}/include")
target_link_libraries(_askme_webrtc_apm PRIVATE "${WEBRTC_APM_LIBRARY}")
target_compile_definitions(_askme_webrtc_apm PRIVATE
  WEBRTC_LIBRARY_IMPL
  WEBRTC_ENABLE_SYMBOL_EXPORT)

if(WIN32)
  target_compile_definitions(_askme_webrtc_apm PRIVATE WEBRTC_WIN NOMINMAX _USE_MATH_DEFINES)
elseif(APPLE)
  target_compile_definitions(_askme_webrtc_apm PRIVATE WEBRTC_MAC WEBRTC_POSIX)
else()
  target_compile_definitions(_askme_webrtc_apm PRIVATE WEBRTC_LINUX WEBRTC_POSIX)
endif()

if(APPLE)
  set_target_properties(_askme_webrtc_apm PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(_askme_webrtc_apm PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _askme_webrtc_apm LIBRARY DESTINATION . RUNTIME DESTINATION .)
install(FILES "${WEBRTC_APM_ROOT}/share/askme-webrtc-apm/SOURCE_LICENSE.txt"
  DESTINATION . RENAME WEBRTC_APM_LICENSE.txt)

# The dependency script deliberately builds shared libraries.  Bundle them
# beside the extension; the release job must still run auditwheel, delocate,
# or delvewheel as documented before publishing a platform wheel.
if(WIN32)
  file(GLOB _runtime_libraries "${WEBRTC_APM_ROOT}/bin/*.dll")
elseif(APPLE)
  file(GLOB _runtime_libraries "${WEBRTC_APM_ROOT}/lib/*.dylib")
else()
  file(GLOB _runtime_libraries
    "${WEBRTC_APM_ROOT}/lib/*.so" "${WEBRTC_APM_ROOT}/lib/*.so.*"
    "${WEBRTC_APM_ROOT}/lib64/*.so" "${WEBRTC_APM_ROOT}/lib64/*.so.*")
endif()
if(NOT _runtime_libraries)
  message(FATAL_ERROR "No WebRTC APM runtime libraries found below ${WEBRTC_APM_ROOT}")
endif()
install(FILES ${_runtime_libraries} DESTINATION .)
