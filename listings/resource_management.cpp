explicit ResourceHandle::ResourceHandle(const uint64_t id) : m_id(id) {}

template <typename T>
ResourceHandle<T>
ResourceManager::RegisterResource(const std::string_view name,
                                  std::unique_ptr<T> resource,
                                  const std::string_view filepath = {}) {
  // Se la risorsa esiste gia', viene rimpiazzata
  if (auto nameIt = m_nameToId.find(name); nameIt != m_nameToId.end()) {
    const uint64_t existingId = nameIt->second;
    // -- Codice omesso --
    return ResourceHandle<T>(existingId);
  }
  // Create una nuova risorsa
  const uint64_t id = GetNextId();
  auto entry = std::make_unique<ResourceEntry>();
  entry->resource = std::move(resource);
  entry->name = name;
  entry->filepath = filepath;
  entry->id = id;
  m_resources[id] = std::move(entry);
  return ResourceHandle<T>(id);
}
