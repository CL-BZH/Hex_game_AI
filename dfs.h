#ifndef DFS_H
#define DFS_H

#include <map>

#include "graph.h"

struct DFS
{
  DFS(const Graph& graph) : graph{graph}
  {
  }

  std::map<unsigned int, bool>& get_map(unsigned int node_id,
                                        std::map<unsigned int, bool>& visited)
  {
    // Init the map
    visited.clear();
    for (unsigned int i{0}; i < graph.V(); ++i) visited[i] = false;
    // Update the map of visited nodes when starting from node with id 'node_id'
    dfs(node_id, visited);
    return visited;
  }

private:
  void dfs(unsigned int v, std::map<unsigned int, bool>& visited)
  {
    // Mark the current node as visited
    visited[v] = true;

    // Recur for all adjacent vertices
    std::vector<std::pair<unsigned int, double>> neighbors;
    graph.get_neighbors(v, neighbors);
    for (auto neighbor : neighbors)
    {
      unsigned int u{neighbor.first};
      if (!visited[u]) dfs(u, visited);
    }
  }

  const Graph& graph;
  std::map<int, bool> visited;
};

#endif  // DFS_H
