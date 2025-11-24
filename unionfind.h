#ifndef UNIONFIND
#define UNIONFIND

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct Subset
{
  unsigned int parent_idx;  // This stores an INDEX, not a node ID
  unsigned int rank;
};

struct UnionFind
{
  explicit UnionFind(const std::vector<unsigned int>& uf_nodes)
  {
    /* Each node defines its own subset */
    for (auto node_id : uf_nodes)
    {
      // Map node_id to index and vice versa
      node_to_index[node_id] = subsets.size();
      index_to_node[subsets.size()] = node_id;
      // Each node starts as its own parent (by index)
      subsets.push_back(Subset{static_cast<unsigned int>(subsets.size()), 0});
    }
  }

  // Copy constructor
  UnionFind(const UnionFind& other)
      : subsets(other.subsets),
        node_to_index(other.node_to_index),
        index_to_node(other.index_to_node)
  {
  }

  friend std::ostream& operator<<(std::ostream& os, const UnionFind& uf)
  {
    for (unsigned int i = 0; i < uf.subsets.size(); ++i)
    {
      os << uf.index_to_node.at(i) << "->" << uf.index_to_node.at(uf.subsets[i].parent_idx) << ",";
    }
    return os;
  }

  /*
   * Get the root of the tree (i.e. the representative) to which belongs the node.
   * A standalone node is the root of its own tree.
   * Path compression is done at the same time.
   */
  unsigned int find(unsigned int node_id)
  {
    unsigned int index = node_to_index.at(node_id);
    unsigned int parent_index = subsets[index].parent_idx;
    unsigned int parent_node_id = index_to_node.at(parent_index);

    // std::cout << "node_id: " << node_id << ", index: "
    //           << index << ", parent_node_id: " << parent_node_id
    //           << ", parent_index: " << parent_index << std::endl;

    // If the parent is not the root, apply path compression
    // We compare NODE IDs, not indices!
    if (parent_node_id != node_id)
    {
      // Recursively find the root
      unsigned int root_node_id = find(parent_node_id);
      unsigned int root_index = node_to_index.at(root_node_id);

      // Path compression: make the root the direct parent
      subsets[index].parent_idx = root_index;

      return root_node_id;
    }

    // This node is the root, return its node_id
    return node_id;
  }

  /*
   * Merge subset (i.e. Union by rank operation):
   * Unites the set that includes i and the set that
   * includes j by rank.
   */
  void merge(unsigned int node1_id, unsigned int node2_id)
  {
    // Find the representative (i.e root node) of each node
    unsigned int root1_id{find(node1_id)};
    unsigned int root2_id{find(node2_id)};

    // Nodes are in the same set, no need to unite
    if (root1_id == root2_id) return;

    // Convert root node IDs back to indices for rank comparison
    unsigned int root1_index = node_to_index.at(root1_id);
    unsigned int root2_index = node_to_index.at(root2_id);

    /* Attach smaller rank tree under root of high rank tree */
    if (subsets[root1_index].rank < subsets[root2_index].rank)
    {
      subsets[root1_index].parent_idx = root2_index;
    }
    else if (subsets[root1_index].rank > subsets[root2_index].rank)
    {
      subsets[root2_index].parent_idx = root1_index;
    }
    else
    {
      // Ranks are equal, attach root2 to root1 and increase root1's rank
      subsets[root2_index].parent_idx = root1_index;
      subsets[root1_index].rank += 1;
    }
  }

  /*
   * Draw the tree structure in a hierarchical format
   */
  void draw(std::ostream& os = std::cout) const
  {
    if (subsets.empty())
    {
      os << "Empty Union-Find structure\n";
      return;
    }

    // Build adjacency list for the tree using node IDs
    std::map<unsigned int, std::set<unsigned int>> children;
    std::set<unsigned int> roots;

    // Initialize roots and build child relationships
    for (unsigned int i = 0; i < subsets.size(); ++i)
    {
      unsigned int node_id = index_to_node.at(i);
      unsigned int parent_index = subsets[i].parent_idx;
      unsigned int parent_id = index_to_node.at(parent_index);

      if (node_id == parent_id)
      {
        roots.insert(node_id);  // This is a root
      }
      else
      {
        children[parent_id].insert(node_id);
      }
    }

    // Draw each connected component (tree)
    for (unsigned int root : roots)
    {
      drawTree(os, root, children, "", true);
      os << "\n";
    }

    // If there are no edges (all nodes are roots), show them as separate components
    if (roots.size() == subsets.size())
    {
      os << "All nodes are separate:\n";
      for (unsigned int root : roots)
      {
        os << root << " ";
      }
      os << "\n";
    }
  }

private:
  /* Collect all subsets - parent fields store INDICES, not node IDs */
  std::vector<Subset> subsets;

  // Mappings between node IDs and internal indices
  std::unordered_map<unsigned int, unsigned int> node_to_index;
  std::unordered_map<unsigned int, unsigned int> index_to_node;

  /*
   * Recursive helper function to draw a tree
   */
  void drawTree(std::ostream& os, unsigned int node_id,
                const std::map<unsigned int, std::set<unsigned int>>& children,
                const std::string& prefix, bool isLast) const
  {
    os << prefix;

    if (isLast)
    {
      os << "└── ";
    }
    else
    {
      os << "├── ";
    }

    // Get the index to access rank information
    unsigned int index = node_to_index.at(node_id);

    // Print node with rank information
    os << node_id << " (rank: " << subsets[index].rank << ")";

    // Mark root nodes
    unsigned int parent_index = subsets[index].parent_idx;
    unsigned int parent_id = index_to_node.at(parent_index);
    if (node_id == parent_id)
    {
      os << " [ROOT]";
    }
    os << "\n";

    // Recursively draw children
    auto it = children.find(node_id);
    if (it != children.end())
    {
      const std::set<unsigned int>& nodeChildren = it->second;
      std::string newPrefix = prefix + (isLast ? "    " : "│   ");

      unsigned int count = 0;
      for (unsigned int child : nodeChildren)
      {
        bool childIsLast = (++count == nodeChildren.size());
        drawTree(os, child, children, newPrefix, childIsLast);
      }
    }
  }
};

#endif  // UNIONFIND
