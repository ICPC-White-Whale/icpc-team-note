# 2021 ACM-ICPC Seoul Regional Team Note

* **Team Name**: White Whale

## KMP

```c++
#include <string>
#include <vector>

using namespace std;

vector<int> getFail(string& str) {
    vector<int> fail(str.size(), 0);
    for (int i = 1, j = 0; i < (int)str.size(); ++i) {
        while (j > 0 && str[i] != str[j]) {
            j = fail[j-1];
        }
        if (str[i] == str[j]) {
            fail[i] = ++j;
        }
    }

    return fail;
}

void KMP(string& para, string& target, vector<int>& fail, vector<int>& found) {
    fail = getFail(target);
    found.clear();

    for (int i = 0, j = 0; i < (int)para.size(); ++i) {
        while (j > 0 && para[i] != target[j]) {
            j = fail[j-1];
        }
        if (para[i] == target[j]) {
            if (j == (int)target.size()-1) {
                found.push_back(i-target.size()+2);
                j = fail[j];
            }
            else {
                j++;
            }
        }
    }
}
```

## ArticulationPoint

```c++
#include <bits/stdc++.h>

using namespace std;

const int SIZE = 100000;
int V, E, dfsn[SIZE], dcnt;
vector<int> adj[SIZE];
vector<pair<int, int>> nodeStack;
vector<vector<pair<int, int>>> bccList;

int dfs(int u, int prv = -1) {
    int ret = dfsn[u] = ++dcnt;
    for (int v : adj[u]) {
        if (v == prv) {
            continue;
        }
        if (dfsn[u] > dfsn[v]) {
            nodeStack.push_back({u, v});
        }
        if (dfsn[v] > 0) {
            ret = min(ret, dfsn[v]);
        } else {
            int tmp = dfs(v, u);
            ret = min(ret, tmp);
            if (tmp >= dfsn[u]) {
                vector<pair<int, int>> curBcc;
                while (!nodeStack.empty() && nodeStack.back() != make_pair(u, v)) {
                    curBcc.push_back(nodeStack.back());
                    nodeStack.pop_back();
                }
                curBcc.push_back(nodeStack.back());
                nodeStack.pop_back();
                bccList.push_back(curBcc);
            }
        }
    }
    return ret;
}

vector<int> getArticulationPoints() {
    for (int i = 0; i < V; i++) {
        if (dfsn[i] == 0) dfs(i);
    }
    vector<int> ret, cnt(V);
    for (auto &curBcc : bccList) {
        set<int> bccNodes;
        for (auto [u, v] : curBcc) {
            bccNodes.insert(u);
            bccNodes.insert(v);
        }
        for (auto u : bccNodes) {
            cnt[u]++;
        }
    }
    for (int i = 0; i < V; i++) {
        if (cnt[i] > 1) {
            ret.push_back(i);
        }
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> V >> E;
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> answer = getArticulationPoints();
    cout << answer.size() << '\n';
    for (auto u : answer) {
        cout << u + 1 << ' ';
    }
}
```

## Trie

```c++
#include <algorithm>
#include <string>

using namespace std;

struct Trie {
    Trie* next[26];

    Trie() {
        fill(next, next + 26, nullptr);
    }

    ~Trie() {
        for (int i = 0; i < 26; i++) {
            if (next[i]) {
                delete next[i];
            }
        }
    }

    void insert(const char *key) {
        if (*key == '\0') {
            return;
        }
        int index = *key - 'a';
        if (!next[index]) {
            next[index] = new Trie;
        }
        next[index]->insert(key + 1);
    }

    void insert(const string& key) {
        insert(key.c_str());
    }
};
```

## ArticulationBridge

```c++
#include <bits/stdc++.h>

using namespace std;
const int SIZE = 100000;
int V, E, low[SIZE], order[SIZE], cnt;
vector<int> adj[SIZE];

void dfs(int u, int p, vector<pair<int, int>> &ret) {
    low[u] = order[u] = ++cnt;
    for (auto v : adj[u]) {
        if (v == p) {
            continue;
        }
        if (order[v] > 0) {
            low[u] = min(low[u], order[v]);
            continue;
        }
        dfs(v, u, ret);
        if (low[v] > order[u]) {
            ret.push_back({min(u, v), max(u, v)});
        }
        low[u] = min(low[u], low[v]);
    }
}

vector<pair<int, int>> getArticulationBridge() {
    vector<pair<int, int>> ret;
    for (int i = 0; i < V; i++) {
        if (order[i]) continue;
        dfs(i, -1, ret);
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> V >> E;
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<pair<int, int>> answer = getArticulationBridge();
    sort(answer.begin(), answer.end());
    cout << answer.size() << '\n';
    for (auto [u, v] : answer) {
        cout << u + 1 << ' ' << v + 1 << '\n';
    }
}
```

