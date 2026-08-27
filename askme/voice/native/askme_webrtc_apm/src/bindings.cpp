#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "api/audio/audio_processing.h"

namespace py = pybind11;

namespace {

constexpr int kNoError = webrtc::AudioProcessing::Error::kNoError;

void ThrowOnApmError(int code, const char* operation) {
  if (code != kNoError) {
    throw std::runtime_error(std::string(operation) + " failed with WebRTC APM error " +
                             std::to_string(code));
  }
}

template <typename T>
py::object OptionalValue(const std::optional<T>& value) {
  return value ? py::cast(*value) : py::none();
}

class AudioProcessingBinding {
 public:
  AudioProcessingBinding(int sample_rate_hz, int channels)
      : frame_samples_(sample_rate_hz / 100 * channels),
        stream_config_(sample_rate_hz, static_cast<size_t>(channels)) {
    if (sample_rate_hz != 8000 && sample_rate_hz != 16000 && sample_rate_hz != 32000 &&
        sample_rate_hz != 48000) {
      throw std::invalid_argument("sample_rate_hz must be a WebRTC int16 native rate");
    }
    if (channels != 1 && channels != 2) {
      throw std::invalid_argument("channels must be mono or stereo");
    }
    processor_ = CreateProcessor();
  }

  void ProcessRender(
      py::array_t<int16_t, py::array::c_style> frame) {
    const auto input = CheckedFrame(frame, "render");
    std::vector<int16_t> processed(static_cast<size_t>(frame_samples_));
    int result;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      result = processor_->ProcessReverseStream(
          static_cast<const int16_t*>(input.ptr), stream_config_, stream_config_,
          processed.data());
    }
    ThrowOnApmError(result, "ProcessReverseStream");
  }

  py::array_t<int16_t> ProcessCapture(
      py::array_t<int16_t, py::array::c_style> frame, int delay_ms) {
    if (delay_ms < 0) {
      throw std::invalid_argument("delay_ms must be non-negative");
    }
    const auto input = CheckedFrame(frame, "capture");
    py::array_t<int16_t> output(frame_samples_);
    auto output_buffer = output.request();
    int delay_result;
    int process_result = kNoError;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      delay_result = processor_->set_stream_delay_ms(delay_ms);
      if (delay_result == kNoError) {
        process_result = processor_->ProcessStream(
            static_cast<const int16_t*>(input.ptr), stream_config_, stream_config_,
            static_cast<int16_t*>(output_buffer.ptr));
      }
    }
    ThrowOnApmError(delay_result, "set_stream_delay_ms");
    ThrowOnApmError(process_result, "ProcessStream");
    return output;
  }

  py::dict Stats() {
    webrtc::AudioProcessingStats stats;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats = processor_->GetStatistics();
    }
    py::dict result;
    result["echo_return_loss_db"] = OptionalValue(stats.echo_return_loss);
    result["echo_return_loss_enhancement_db"] =
        OptionalValue(stats.echo_return_loss_enhancement);
    result["residual_echo_likelihood"] = OptionalValue(stats.residual_echo_likelihood);
    return result;
  }

  void Reset() {
    auto replacement = CreateProcessor();
    std::lock_guard<std::mutex> lock(mutex_);
    processor_ = std::move(replacement);
  }

 private:
  py::buffer_info CheckedFrame(
      const py::array_t<int16_t, py::array::c_style>& frame,
      const char* stream_name) const {
    auto buffer = frame.request();
    if (buffer.ndim != 1 || buffer.size != frame_samples_) {
      throw std::invalid_argument(std::string(stream_name) +
                                  " must contain exactly one interleaved 10 ms frame");
    }
    return buffer;
  }

  rtc::scoped_refptr<webrtc::AudioProcessing> CreateProcessor() const {
    auto processor = webrtc::AudioProcessingBuilder().Create();
    if (!processor) {
      throw std::runtime_error("AudioProcessingBuilder returned null");
    }

    webrtc::AudioProcessing::Config config;
    config.echo_canceller.enabled = true;
    config.echo_canceller.mobile_mode = false;
    config.high_pass_filter.enabled = true;
    config.noise_suppression.enabled = true;
    config.noise_suppression.level =
        webrtc::AudioProcessing::Config::NoiseSuppression::kHigh;
    config.gain_controller1.enabled = false;
    config.gain_controller2.enabled = false;
    processor->ApplyConfig(config);

    webrtc::ProcessingConfig processing_config;
    processing_config.input_stream() = stream_config_;
    processing_config.output_stream() = stream_config_;
    processing_config.reverse_input_stream() = stream_config_;
    processing_config.reverse_output_stream() = stream_config_;
    ThrowOnApmError(processor->Initialize(processing_config), "Initialize");
    return processor;
  }

  py::ssize_t frame_samples_;
  webrtc::StreamConfig stream_config_;
  rtc::scoped_refptr<webrtc::AudioProcessing> processor_;
  std::mutex mutex_;
};

}  // namespace

PYBIND11_MODULE(_askme_webrtc_apm, module) {
  module.doc() = "Pinned WebRTC Audio Processing v2.1 binding for AskMe";
  py::class_<AudioProcessingBinding>(module, "AudioProcessing")
      .def(py::init<int, int>(), py::arg("sample_rate_hz"), py::arg("channels"))
      .def("process_render", &AudioProcessingBinding::ProcessRender, py::arg("frame"))
      .def("process_capture", &AudioProcessingBinding::ProcessCapture, py::arg("frame"),
           py::arg("delay_ms"))
      .def("stats", &AudioProcessingBinding::Stats)
      .def("reset", &AudioProcessingBinding::Reset);
}
