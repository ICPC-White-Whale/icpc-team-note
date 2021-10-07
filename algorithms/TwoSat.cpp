#include <bits/stdc++.h>
using namespace std;

const int MAX = 20001;
int N, M, dfsn[MAX], finished[MAX], dfsCnt, sccCnt;
vector<int> adj[MAX];
vector<int> st;

int getIdx(int x) {
    // {-N <= i <= N} to {0 <= i < 2N}
    // !x는 홀수 인덱스, x는 짝수 인덱스
    if (x < 0) return (-x - 1) * 2 + 1;
    else return (x - 1) * 2;
}

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

bool isSatisfied() {
    for (int i = 0; i < N * 2; i++) {
        if (!dfsn[i]) dfs(i);
    }
    vector<vector<int>> sccList(sccCnt);
    for (int i = 0; i < N * 2; i++) {
        sccList[finished[i] - 1].push_back(i);
    }
    for (int i = 1; i <= N; i++) {
        if (finished[getIdx(i)] == finished[getIdx(-i)]) {
            return false;
        }
    }
    return true;
}

vector<int> getTwoSat() {
    vector<int> ret(N, -1);
    vector<pair<int, int>> ts(2 * N);
    for (int i = 0; i < N * 2; i++) {
        ts[i].first = finished[i];
        ts[i].second = i;
    }
    sort(ts.begin(), ts.end(), greater<>());
    // 위상정렬 순서대로
    for (auto [_, i] : ts) {
        if (ret[i / 2] >= 0) continue;
        ret[i / 2] = i % 2; // ~x는 true, x는 false 할당
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> N >> M;
    for (int i = 0; i < M; i++) {
        int a, b;
        cin >> a >> b;
        adj[getIdx(-a)].push_back(getIdx(b));
        adj[getIdx(-b)].push_back(getIdx(a));
    }
    if (!isSatisfied()) {
        cout << 0 << '\n';
    } else {
        cout << 1 << '\n';
        vector<int> twoSat = getTwoSat();
        for (auto isTrue : twoSat) {
            cout << isTrue << ' ';
        }
    }
}

