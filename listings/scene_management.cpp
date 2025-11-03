class Node {
public:
  void AddChild(std::unique_ptr<Node> child);
  bool RemoveChild(const Node *child);
  // Tree iteration
  void ForEachChild(const std::function<void(Node *)> &func,
                    const bool recursive = false);
  void ForEachChild(const std::function<void(const Node *)> &func,
                    const bool recursive = false) const;

private:
  Node *m_parent;
  std::vector<std::unique_ptr<Node>> m_children;
  std::vector<std::unique_ptr<Component>> m_components;
};
