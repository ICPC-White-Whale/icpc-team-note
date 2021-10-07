#include <bits/stdc++.h>

using namespace std;

const int MAX = 10000;
int N, M, dfsn[MAX], finished[MAX], dfsCnt, sccCnt;
vector<int> adj[MAX];
vector<int> st;

int dfs(int u) {
    dfsn[u] = ++dfsCnt;
    st.push_back(u);
    int ret = dfsn[u];
    for (auto v : adj[u]) {
        if (!dfsn[v]) {
            ret = min(ret, dfs(v));
        }
        else if (!finished[v]) {
            ret = min(ret, dfsn[v]);
        }
    }
    if (ret == dfsn[u]) {
        sccCnt++;
        while (true) {
            int t = st.back();
            st.pop_back();
            finished[t] = sccCnt;
            if (t == u) break;
        }
    }
    return ret;
}

vector<vector<int>> getSccList() {
    for (int i = 0; i < N; i++) {
        if (!dfsn[i]) dfs(i);
    }
    vector<vector<int>> sccList(sccCnt);
    for (int i = 0; i < N; i++) {
        sccList[finished[i] - 1].push_back(i);
    }
    return sccList;
}

