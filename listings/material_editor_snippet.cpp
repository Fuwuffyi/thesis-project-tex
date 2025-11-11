void MaterialEditor::DrawTextureSlotEditor(
    IMaterial *const material, const std::string_view textureName,
    const std::string_view displayName) const {

  const TextureHandle currentTex = material->GetTexture(textureName);
  const ITexture *const texture = m_resourceManager->GetTexture(currentTex);

  // Drag target for texture assignment
  if (ImGui::BeginDragDropTarget()) {
    if (const ImGuiPayload *const payload =
            ImGui::AcceptDragDropPayload(kTexturePayload.data())) {
      const std::string textureNameStr{
          static_cast<const char *>(payload->Data)};
      const TextureHandle newTex =
          m_resourceManager->GetTextureHandle(textureNameStr);
      if (newTex.IsValid()) {
        material->SetTexture(textureName, newTex);
      }
    }
    ImGui::EndDragDropTarget();
  }
}
