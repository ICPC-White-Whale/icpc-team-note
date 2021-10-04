#include <bits/stdc++.h>

using namespace std;
const int MAX_V = 1005;

struct edge {
    int to, cap, f, cost;
    edge *dual;
    edge(int to1, int cap1, int cost1): to(to1), cap(cap1), cost(cost1), f(0), dual(nullptr) {}
    edge(): edge(-1, 0, 0) {}

    int spare() {
        return cap - f;
    }

    void addFlow(int f1) {
        f += f1;
        dual->f -= f1;
    }
};

int maxFlow, minCost;
vector<edge*> adj[MAX_V];

void addEdge(int from, int to, int cap, int cost) {
    edge *e1 = new edge(to, cap, cost), *e2 = new edge(from, 0, -cost);
    e1->dual = e2;
    e2->dual = e1;
    adj[from].push_back(e1);
    adj[to].push_back(e2);
}

void mcmf(int s, int e) {
    while (true) {
        vector<int> prev(MAX_V, -1), dist(MAX_V, 1e9);
        vector<bool> visited(MAX_V, false);
        vector<edge*> path(MAX_V);
        queue<int> q;
        q.push(s);
        dist[s] = 0;
        visited[s] = true;

        while(!q.empty()){
            int u = q.front();
            q.pop();
            visited[u] = false;
            for(edge *e: adj[u]){
                int v = e->to;
                if(e->spare() > 0 && dist[v] > dist[u] + e->cost){
                    dist[v] = dist[u] + e->cost;
                    prev[v] = u;
                    path[v] = e;
                    if (!visited[v]) {
                        q.push(v);
                        visited[v] = true;
                    }
                }
            }
        }
        if(prev[e] == -1) break;

        int flow = 1e9;
        for(int i = e; i != s; i = prev[i]) {
            flow = min(flow, path[i]->spare());
        }
        for(int i = e; i != s; i = prev[i]) {
            minCost += path[i]->cost * flow;
            path[i]->addFlow(flow);
        }
        maxFlow += flow;
    }
}