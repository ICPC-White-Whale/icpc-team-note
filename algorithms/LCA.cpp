#include <bits/stdc++.h>
using namespace std;

const int MAX_D = 17;
const int MAXN = 100000;

int N, p[MAXN][MAX_D], depth[MAXN];
vector<int> adj[MAXN];
bool vst[MAXN];

void dfs(int u, int d) {
    vst[u] = true;
    depth[u] = d;
    for (int v : adj[u])
        if (!vst[v]) {
            p[v][0] = u;
            dfs(v, d + 1);
        }
}

void constructLca() {
    dfs(0, 0);
    for (int j = 1; j < MAX_D; j++) {
        for (int i = 1; i < N; i++) {
            p[i][j] = p[p[i][j - 1]][j - 1];
        }
    }
}

int findLca(int u, int v) {
    // Make u have u higher depth
    if (depth[u] < depth[v]) swap(u, v);

    // Elevate u to the depth of v
    int depth_diff = depth[u] - depth[v];
    for (int j = MAX_D - 1; j >= 0; j--) {
        if (depth_diff & (1 << j)) {
            u = p[u][j];
        }
    }

    if (u == v) return u;

    for (int j = MAX_D - 1; j >= 0; j--) {
        if (p[u][j] != p[v][j]) {
            u = p[u][j];
            v = p[v][j];
        }
    }

    return p[u][0];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> N;
    for (int i = 0; i < N - 1; i++) {
        int x, y;
        cin >> x >> y;
        x--; y--;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
    constructLca();
}