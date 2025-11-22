#ifndef UNIONFIND
#define UNIONFIND

struct Subset
{
  unsigned int parent;
  unsigned int rank;
};

struct UnionFind
{
  explicit UnionFind(unsigned int nodes = 0)
  {
    /* Each node defines its own subset */
    for (unsigned int i = 0; i < nodes; ++i) subsets.push_back(Subset{i, 0});
  }

  friend std::ostream& operator<<(std::ostream& os, const UnionFind& uf)
  {
    for (auto node : uf.subsets) os << node.parent << ',';
    return os;
  }

  /*
   * Get the root of the tree to which belongs the node.
   * Path compression is done at the same time.
   */
  unsigned int find(unsigned int node)
  {
    unsigned int parent{subsets[node].parent};
    if (parent != node) subsets[node].parent = find(parent);
    return subsets[node].parent;
  }

  /* Merge subset (i.e. Union operation) */
  void merge(unsigned int node1, unsigned int node2)
  {
    unsigned int root1{find(node1)};
    unsigned int root2{find(node2)};

    if (root1 == root2) return;

    /* Attach smaller rank tree under root of high rank tree */
    if (subsets[root1].rank < subsets[root2].rank)
    {
      subsets[root1].parent = root2;
    }
    else
    {
      subsets[root2].parent = root1;
      if (subsets[root1].rank == subsets[root2].rank) subsets[root1].rank += 1;
    }
  }

private:
  /* Collect all subsets */
  std::vector<Subset> subsets;
};

#endif  // UNIONFIND
