void PerformanceGUI::DrawPerformanceGraph() noexcept {
  const auto &history = s_history.GetHistory();
  auto validFrameTimes =
      history | std::views::filter([](float ft) { return ft > 0.0f; });

  if (std::ranges::empty(validFrameTimes)) {
    ImGui::Text("No frame time data available");
    return;
  }

  std::vector<float> frameTimesVec;
  std::ranges::copy(validFrameTimes, std::back_inserter(frameTimesVec));

  const auto [minTime, maxTime] = std::ranges::minmax_element(frameTimesVec);
  const float scaleMin = std::max(0.0f, *minTime - 1.0f);
  const float scaleMax = *maxTime + 1.0f;

  ImGui::PlotLines("Frame Times", frameTimesVec.data(),
                   static_cast<int>(frameTimesVec.size()), 0,
                   overlayText.c_str(), scaleMin, scaleMax, kGraphSize);
}
