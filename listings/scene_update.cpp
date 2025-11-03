void Scene::UpdateTransforms() {
  if (m_rootNode) {
    m_rootNode->UpdateWorldTransform(
        true); // Aggiorna world transform ricorsivamente
  }
}

void Scene::UpdateScene(const float deltaTime) {
  UpdateTransforms();
  ForEachNode([deltaTime](Node *node) {
    auto particleComponent = node->GetComponent<ParticleSystemComponent>();
    if (particleComponent) {
      particleComponent->Update(deltaTime,
                                node->GetWorldTransform()->GetPosition());
    }
  });
}
