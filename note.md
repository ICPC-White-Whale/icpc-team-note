# 2021 ACM-ICPC Seoul Regional Team Note

* **Team Name**: White Whale

## Aho-Corasick

```c++
#include <algorithm>
#include <vector>
#include <queue>
#include <string>

using namespace std;

struct Trie {
    Trie *next[26];
    Trie *fail;
    // 실제 매칭된 문자열이 필요하다면 아래 정의 사용
    // vector<string> outputs;
    // 매칭 여부만 필요하다면
    bool matched = false;

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

    void insert(string &str, int start) {
        if ((int)str.size() <= start) {
            // 실제 매칭된 문자열이 필요하다면 아래 정의 사용
            //outputs.push_back(str);
            // 매칭 여부만 필요하다면
            matched = true;
            return;
        }
        int nextIdx = str[start] - 'a';
        if (!next[nextIdx]) {
            next[nextIdx] = new Trie;
        }
        next[nextIdx]->insert(str, start + 1);
    }
};

void buildFail(Trie *root) {
    queue<Trie *> q;
    root->fail = root;
    q.push(root);

    while (!q.empty()) {
        Trie *current = q.front();
        q.pop();

        for (int i = 0; i < 26; i++) {
            Trie *next = current->next[i];

            if (!next) {
                continue;
            } else if (current == root) {
                next->fail = root;
            } else {
                Trie *dest = current->fail;
                while (dest != root && !dest->next[i]) {
                    dest = dest->fail;
                }
                if (dest->next[i]) {
                    dest = dest->next[i];
                }
                next->fail = dest;
            }

            // 실제 매칭된 문자열이 필요하다면 아래 정의 사용
            // if (next->fail->outputs.size() > 0) {
            //     next->outputs.insert(next->outputs.end(), current->outputs.begin(), current->outputs.end());
            // }
            // 매칭 여부만 필요하다면
            if (next->fail->matched) {
                next->matched = true;
            }
            q.push(next);
        }
    }
}

bool find(Trie *root, string &query) {
    Trie *current = root;
    bool result = false;

    for (int c = 0; c < (int)query.size(); c++) {
        int nextIdx = query[c] - 'a';
        while (current != root && !current->next[nextIdx]) {
            current = current->fail;
        }
        if (current->next[nextIdx]) {
            current = current->next[nextIdx];
        }
        if (current->matched) {
            result = true;
            break;
        }
    }
    return result;
}
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

## ConvexHull and RotatingCalipers

```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long Long;
typedef pair<Long, Long> point;
typedef pair<Long, pair<Long, Long>> dot;

int N;
vector<dot> dots;

int ccw(point &a, point &b, point &c) {
    point ab;
    ab.first = b.first - a.first;
    ab.second = b.second - a.second;
    point bc;
    bc.first = c.first - b.first;
    bc.second = c.second - b.second;
    Long ret = ab.first * bc.second - ab.second * bc.first;
    ret = -ret;
    if (ret > 0) return 1;
    else if (ret == 0) return 0;
    else return -1;
}

Long getDistance(point &A, point &B) {
    Long dx = A.first - B.first;
    Long dy = A.second - B.second;
    return dx * dx + dy * dy;
}

bool comp(dot &A, dot &B) {
    int cw = ccw(dots[0].second, A.second, B.second);
    if (cw > 0) return true;
    if (cw < 0) return false;
    else if (A.first < B.first) return true; // dist A < dist B (기준 점에서 거리)
    return false;
}

void setSlope(dot &P0) {
    for (int i = 1; i < N; i++) {
        dots[i].first = getDistance(P0.second, dots[i].second);
    }
    sort(dots.begin() + 1, dots.end(), comp);
}

vector<point> getConvexHull() {
    sort(dots.begin(), dots.end());
    setSlope(dots[0]);
    vector<point> ch;
    ch.push_back(dots[0].second);
    ch.push_back(dots[1].second);
    for (int i = 2; i < (int)dots.size(); i++) {
        while ((int)ch.size() >= 2 && ccw(ch[(int)ch.size() - 2], ch[(int)ch.size() - 1], dots[i].second) <= 0) {
            ch.pop_back();
        }
        ch.push_back(dots[i].second);
    }
    return ch;
}

Long getVectorCross(const point &p1, const point &p2) {
    return p1.second * p2.first - p1.first * p2.second;
}

pair<int, int> getMaxDistPair(vector<point> &ch) {
    int l = 0, r = 0, M = (int)ch.size();
    for (int i = 0; i < M; i++) {
        if (ch[i].second < ch[l].second) l = i;
        if (ch[i].second > ch[r].second) r = i;
    }
    long long maxDist = 0;
    int maxDistL = -1, maxDistR = -1;
    for (int i = 0; i <= M; i++) {
        if (maxDist < getDistance(ch[l], ch[r])) {
            maxDist = getDistance(ch[l], ch[r]);
            maxDistL = l;
            maxDistR = r;
        }
        point leftVector = {ch[l].second - ch[(l + 1) % M].second,
                            ch[l].first - ch[(l + 1) % M].first};
        point rightVector = {ch[r].second - ch[(r + 1) % M].second,
                             ch[r].first - ch[(r + 1) % M].first};

        if (getVectorCross(leftVector, rightVector) > 0) {
            l = (l + 1) % M;
        } else {
            r = (r + 1) % M;
        }
    }
    return {maxDistL, maxDistR};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    int T;
    cin >> T;
    while (T--) {
        cin >> N;
        dots.clear();
        dots.resize(N);
        for (int i = 0; i < N; i++) {
            Long Dx, Dy;
            cin >> Dx >> Dy;
            dots[i].second.first = Dy;
            dots[i].second.second = Dx;
        }
        vector<point> convexHull = getConvexHull();
        auto[maxDistL, maxDistR] = getMaxDistPair(convexHull);
        cout << convexHull[maxDistL].second << ' ' << convexHull[maxDistL].first << ' ';
        cout << convexHull[maxDistR].second << ' ' << convexHull[maxDistR].first << '\n';
    }
}
```

## Dinic

```c++
#include <bits/stdc++.h>
using namespace std;

class Dinic {
    struct Edge {
        int v, cap, rev;
        Edge(int v, int cap, int rev) : v(v), cap(cap), rev(rev) {}
    };

    const int INF = 2e9;
    int MAX_V;
    int S, E;  // source, sink
    vector<vector<Edge>> adj;
    vector<int> level, work;

    bool bfs() {
        fill(level.begin(), level.end(), -1);
        queue<int> qu;
        level[S] = 0;
        qu.push(S);
        while (qu.size()){
            int here = qu.front();
            qu.pop();
            for (auto i : adj[here]) {
                int there = i.v;
                int cap = i.cap;
                if (level[there] == -1 && cap > 0) {
                    level[there] = level[here] + 1;
                    qu.push(there);
                }
            }
        }
        return level[E] != -1;
    }

    int dfs(int here, int crtcap) {
        if (here == E) return crtcap;
        for (int &i = work[here]; i < int(adj[here].size()); i++) {
            int there = adj[here][i].v;
            int cap = adj[here][i].cap;
            if (level[here] + 1 == level[there] && cap > 0) {
                int c = dfs(there, min(crtcap, cap));
                if (c > 0) {
                    adj[here][i].cap -= c;
                    adj[there][adj[here][i].rev].cap += c;
                    return c;
                }
            }
        }
        return 0;
    }

public:
    Dinic(int MAX_V) : MAX_V(MAX_V) {
        adj.resize(MAX_V);
        level.resize(MAX_V);
        work.resize(MAX_V);
    }
    void addEdge(int s, int e, int c) {
        adj[s].emplace_back(e, c, (int)adj[e].size());
        adj[e].emplace_back(s, 0, (int)adj[s].size() - 1);
    }

    int getMaxFlow(int s, int e) {
        S = s, E = e;
        int res = 0;
        while (bfs()) {
            fill(work.begin(), work.end(), 0);
            while (1) {
                int flow = dfs(S, INF);
                if (!flow) break;
                res += flow;
            }
        }
        return res;
    }
};

```

## HopcroftKarp

```c++
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
```

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

vector<int> KMP(string& para, string& target) {
    vector<int> fail = getFail(target);
    vector<int> found;

    for (int i = 0, j = 0; i < (int)para.size(); ++i) {
        while (j > 0 && para[i] != target[j]) {
            j = fail[j-1];
        }
        if (para[i] == target[j]) {
            if (j == (int)target.size()-1) {
                found.push_back(i-target.size()+2);
                j = fail[j];
            } else {
                j++;
            }
        }
    }
    return found;
}
```

## MCMF

```c++
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
```

## SCC

```c++
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


```

## Suffix Array and LCP

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

void extractLCP(const string& str, vector<int>& sa, vector<int>& group, vector<int>& lcp) {
    int n = str.size();
    sa.resize(n);
    group.resize(n);
    lcp.resize(n-1);

    iota(sa.begin(), sa.end(), 0);
    for (int i = 0; i < sa.size(); ++i) {
        group[i] = str[i];
    }

    for (int d = 1; ; d <<= 1) {
        auto compareSuffix = [&](int s1, int s2) -> bool {
            if (group[s1] != group[s2]) {
                return group[s1] < group[s2];
            } else if (s1+d < n && s2+d < n) {
                return group[s1+d] < group[s2+d];
            } else {
                return s1 > s2;
            }
        };
        sort(sa.begin(), sa.end(), compareSuffix);

        vector<int> newGroup(n, 0);
        for (int i = 0; i < n-1; ++i) {
            newGroup[i+1] = newGroup[i] + (compareSuffix(sa[i], sa[i+1]) ? 1 : 0);
        }
        for (int i = 0; i < n; ++i) {
            group[sa[i]] = newGroup[i];
        }

        if (newGroup.back() == n-1) {
            break;
        }
    }

    for (int i = 0, k = 0; i < n; ++i, k = max(k-1, 0)) {
        if (group[i] != n-1) {
            for (int j = sa[group[i]+1]; str[i+k] == str[j+k]; ++k);
            lcp[group[i]] = k;
        }
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

## TwoSat

```c++
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


```

