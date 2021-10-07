#include <bits/stdc++.h>

using namespace std;
const int MAX = 10000;
const int INF = 1e9;

int N, M, A[MAX], B[MAX], dist[MAX];
bool used[MAX], checkA[MAX], checkB[MAX];
vector<int> adj[MAX];
vector<int> leftCover, rightCover;

void bfs() {
    queue<int> q;
    for (int i = 0; i < N; i++) {
        if (!used[i]) {
            dist[i] = 0;
            q.push(i);
        } else dist[i] = INF;
    }
    while (!q.empty()) {
        int a = q.front();
        q.pop();
        for (int b: adj[a]) {
            if (B[b] != -1 && dist[B[b]] == INF) {
                dist[B[b]] = dist[a] + 1;
                q.push(B[b]);
            }
        }
    }
}

bool dfs(int a) {
    for (int b: adj[a]) {
        if (B[b] == -1 || (dist[B[b]] == dist[a] + 1 && dfs(B[b]))) {
            used[a] = true;
            A[a] = b;
            B[b] = a;
            return true;
        }
    }
    return false;
}

int hopcroftKarp() {
    int match = 0;
    fill(A, A + MAX, -1);
    fill(B, B + MAX, -1);
    while (true) {
        bfs();
        int flow = 0;
        for (int i = 0; i < N; i++)
            if (!used[i] && dfs(i)) {
                flow++;
            }
        if (flow == 0) break;
        match += flow;
    }
    return match;
}

void dfsCover(int x) {
    if (checkA[x]) {
        return;
    }
    checkA[x] = true;
    for (auto i : adj[x]) {
        checkB[i] = true;
        dfsCover(B[i]);
    }
}

void getCover() {
    for (int i = 0; i < N; i++) {
        if (A[i] == -1) {
            dfsCover(i);
        }
    }
    for (int i = 0; i < N; i++) {
        if (!checkA[i]) {
            leftCover.push_back(i);
        }
    }
    for (int i = 0; i < M; i++) {
        if (checkB[i]) {
            rightCover.push_back(i);
        }
    }
}

int main() {
    cin >> N >> M;
    for (int i = 0; i < N; i++) {
        int k;
        cin >> k;
        while (k--) {
            int b;
            cin >> b;
            adj[i].push_back(b - 1);
        }
    }
    int max_match = hopcroftKarp();
    getCover();
}