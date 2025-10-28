std::unique_ptr<IRenderer> RendererFactory::CreateRenderer(const GraphicsAPI api, Window* const win) {
   switch (api) {
      case GraphicsAPI::OpenGL:
         return std::make_unique<GLRenderer>(win);
      case GraphicsAPI::Vulkan:
         return std::make_unique<VulkanRenderer>(win);
      default:
         throw std::runtime_error("Unsupported Graphics API");
   }
}
