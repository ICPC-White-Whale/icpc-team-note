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

## Centroid

```c++
#include <vector>
#define MAX 1001

using namespace std;

int sz[MAX];
std::vector<int> adj[MAX];

int getSz(int here,int dad) {
    sz[here] = 1;
    for (auto there : adj[here]){
        if (there == dad) {
            continue;
        }
        sz[here]+=getSz(there,here);
    }
    return sz[here];
}
 
int get_centroid(int here, int dad, int cap) {
    //cap <---- (tree size)/2
    for(auto there : adj[here]){
        if (there == dad) {
            continue;
        }
        if(sz[there] > cap) {
            return get_centroid(there,here,cap);
        }
    }
    return here;
}

int main() {
    int root = 1;
    getSz(root, -1);
    int center = get_centroid(1, -1, sz[root]/2);
    return 0;
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
    point ab = {b.first - a.first, b.second - a.second};
    point bc = {c.first - b.first, c.second - b.second};
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

## Divide and Conquer

```c++
// DnQ Optimization을 적용하기 위한 조건

/*
조건 1: dp[t][i] = min_{k<i} (dp[t-1][k] + C[k][i])
*/

/*
조건 2: 아래 두 조건들 중 적어도 하나를 만족

    a)  A[t][i]를 dp[t][i]를 만족시키는 최소의 k라고 할 때 아래 부등식을 만족
        A[t][i] <= A[t][i+1]
    
    b)  비용 C가 a<=b<=c<=d인 a, b, c, d에 대하여
        사각부등식 C[a][c] + C[b][d] <= C[a][d] + C[b][c] 를 만족
*/

// 위 두 조건이 만족될 경우, O(KN log N)으로 해결가능

typedef long long Long;

int L, G;
Long Ci[8001];
Long sum[8001];

Long dp[801][8001], properK[801][8001];

// 문제에 맞게 Cost 정의
Long getCost(Long a, Long b) {
    return (sum[b] - sum[a - 1]) * (b - a + 1);
}

// dp[t][i] = min_{k<i} (dp[t-1][k] + C[k][i]) 꼴의 문제를 풀고자 할 때,
// 아래 함수는 dp[t][l~r]을 채운다.
void Find(int t, int l, int r, int p, int q) {
    if (l > r) {
        return;
    }
    int mid = (l + r) >> 1;
    dp[t][mid] = -1;
    properK[t][mid] = -1;

    for (int k = p; k <= q; ++k) {
        Long current = dp[t - 1][k] + getCost(k+1, mid);
        if (dp[t][mid] == -1 || dp[t][mid] > current) {
            dp[t][mid] = current;
            properK[t][mid] = k;
        }
    }

    Find(t, l, mid - 1, p, properK[t][mid]);
    Find(t, mid + 1, r, properK[t][mid], q);
}
```

## Extended Euclidian Algorithm

```c++
#include <vector>
#include <iostream>

using namespace std;

struct INFO {
    int gcd;
    int s;
    int t;
};

vector<int> s, t, r, q;

INFO xGCD(int a, int b) {
    s = { 1,0 };
    t = { 0,1 };
    r = { a,b };

    while (1) {
        int r2 = r[r.size() - 2];
        int r1 = r[r.size() - 1];
        q.push_back(r2 / r1);
        r.push_back(r2 % r1);
        
        if (r[r.size() - 1] == 0) {
            break;
        }
        int s2 = s[s.size() - 2];
        int s1 = s[s.size() - 1];

        int t2 = t[t.size() - 2];
        int t1 = t[t.size() - 1];

        int q1 = q[q.size() - 1];
        s.push_back(s2 - s1 * q1);
        t.push_back(t2 - t1 * q1);
    }
    // return gcd, s, t
    INFO ret = { r[r.size() - 2], s[s.size() - 1], t[t.size() - 1] };
    s.clear(), t.clear(), r.clear(), q.clear();
    return ret;
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    if (b > a) {
        swap(a, b);
    }

    INFO ret = xGCD(a, b);
    printf("gcd :: %d s :: %d t :: %d", ret.gcd, ret.s, ret.t);
    
    return 0;
}
// 출처: https://www.crocus.co.kr/1232 [Crocus]
```

## FFT

```c++
#include <iostream>
#include <vector>
#include <complex>
using namespace std;

typedef long long Long;
const double pi = 3.14159265358979323846264;

typedef complex<double> base;

void fft(vector<base> &a, bool inv){
    int n = a.size(), j = 0;
    vector<base> roots(n/2);
    for(int i=1; i<n; i++){
        int bit = (n >> 1);
        while(j >= bit){
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if(i < j) swap(a[i], a[j]);
    }
    double ang = 2 * acos(-1) / n * (inv ? -1 : 1);
    for(int i=0; i<n/2; i++){
        roots[i] = base(cos(ang * i), sin(ang * i));
    }

    for(int i=2; i<=n; i<<=1){
        int step = n / i;
        for(int j=0; j<n; j+=i){
            for(int k=0; k<i/2; k++){
                base u = a[j+k], v = a[j+k+i/2] * roots[step * k];
                a[j+k] = u+v;
                a[j+k+i/2] = u-v;
            }
        }
    }
    if(inv) for(int i=0; i<n; i++) a[i] /= n; // skip for OR convolution.
}

vector<Long> multiply(vector<Long> &v, vector<Long> &w){
    vector<base> fv(v.begin(), v.end()), fw(w.begin(), w.end());
    int n = 2;
    while(n < v.size() + w.size()) n <<= 1;
    fv.resize(n); fw.resize(n);
    fft(fv,0); fft(fw,0);
    for(int i=0; i<n; i++) fv[i] *= fw[i];
    fft(fv,1);
    vector<Long> ret(n);
    for(int i=0; i<n; i++) ret[i] = (Long)round(fv[i].real());
    return ret;
}

vector<Long> multiply(vector<Long> &v, vector<Long> &w, Long mod){
    int n = 2;
    while(n < v.size() + w.size()) n <<= 1;
    vector<base> v1(n), v2(n), r1(n), r2(n);
    for(int i=0; i<v.size(); i++){
        v1[i] = base(v[i] >> 15, v[i] & 32767);
    }
    for(int i=0; i<w.size(); i++){
        v2[i] = base(w[i] >> 15, w[i] & 32767);
    }
    fft(v1,0);
    fft(v2,0);
    for(int i=0; i<n; i++){
        int j = (i ? (n - i) : i);
        base ans1 = (v1[i] + conj(v1[j])) * base(0.5,0);
        base ans2 = (v1[i] - conj(v1[j])) * base(0,-0.5);
        base ans3 = (v2[i] + conj(v2[j])) * base(0.5,0);
        base ans4 = (v2[i] - conj(v2[j])) * base(0,-0.5);
        r1[i] = (ans1 * ans3) + (ans1 * ans4) * base(0,1);
        r2[i] = (ans2 * ans3) + (ans2 * ans4) * base(0,1);
    }
    fft(r1,1);
    fft(r2,1);
    vector<Long> ret(n);
    for(int i=0; i<n; i++){
        Long av = (Long)round(r1[i].real());
        Long bv = (Long)round(r1[i].imag()) + (Long)round(r2[i].real());
        Long cv = (Long)round(r2[i].imag());
        av %= mod, bv %= mod, cv %= mod;
        ret[i] = (av << 30) + (bv << 15) + cv;
        ret[i] %= mod;
        ret[i] += mod;
        ret[i] %= mod;
    }
    return ret;
}

```

## Fermat's Little Theorem

```c++
long long pow(long long n, long long r, int MOD) {
    long long ret = 1;
    for (; r; r >>= 1) {
        if (r & 1) ret = ret * n % MOD;
        n = n * n % MOD;
    }
    return ret;
}
/* A * B^(p−2)  (mod p)
long long ans = fact[a];
ans = ans * pow(fact[b], MOD - 2, MOD) % MOD;
ans = ans * pow(fact[a - b], MOD - 2, MOD) % MOD;
*/
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

## IndexTree

```c++
#include <bits/stdc++.h>
using namespace std;

const int SIZE = 4194304;

struct SegTree{
    int size, start;
    long long arr[SIZE];

    SegTree(int n): size(n){
        start = 1;
        while (start < size) start *= 2;
        memset(arr, 0, sizeof(arr));
    }

    void set(int here, long long val) {
        // 0-index, require prepare()
        arr[start + here] = val;
    }

    void prepare(){
        for (int i = start - 1; i; i--) {
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
        }
    }

    void update(int here, long long val){
        // 0-index
        here += start;
        arr[here] += val;
        while (here){
            here /= 2;
            arr[here] = arr[here * 2] + arr[here * 2 + 1];
        }
    }

    long long sum(int l, int r){
        // [l, r], 0-index
        l += start;
        r += start;
        long long ret = 0;
        while (l <= r){
            if (l % 2 == 1) ret += arr[l++];
            if (r % 2 == 0) ret += arr[r--];
            l /= 2; r /= 2;
        }
        return ret;
    }

    int search(int k) {
        // search kth number, k >= 1
        int pos = 1;
        while (pos < start) {
            if (k <= arr[pos * 2]) {
                pos *= 2;
            }
            else {
                k -= arr[pos * 2];
                pos = pos * 2 + 1;
            }
        }
        return pos - start;
    }
};
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

## Knuth Optimization

```c++
// Knuth Optimization을 적용하기 위한 조건

/*
조건 1: dp[i][j] = min_{i<k<j} (dp[i][k] + dp[k][j]) + C[i][j]
조건 2: C[a][c] + C[b][d] <= C[a][d] + C[b][c] (a <= b <= c <= d)
조건 3: C[b][c] <= C[a][d] (a <= b <= c <= d)
*/

// 위 세 조건이 만족될 경우, O(N^2)으로 해결가능

typedef long long Long;
const Long INF = 1LL<<32;

int data[1003];
Long d[1003][1003];
int p[1003][1003];

int getCost(int left, int right) {
    // define your cost function here
}

// data의 [left, right]에 값을 채우고 아래 함수를 실행하면 d에 dp값이 채워진다.
void doKnuthOpt(int left, int right) {
    for (int i = left; i <= right; i++) {
		d[i][i] = 0, p[i][i] = i;
		for (int j = i + 1; j <= right; j++) {
			d[i][j] = 0, p[i][j] = i;
        }
	}

	for (int l = 2; l <= right-left+1; l++)  {
		for (int i = left; i + l <= right; i++) {
			int j = i + l;
			d[i][j] = INF;
			for (int k = p[i][j - 1]; k <= p[i + 1][j]; k++) {
                int current = d[i][k] + d[k][j] + getCost(i, j);
				if (d[i][j] > current) {
					d[i][j] = current;
					p[i][j] = k;
				}
			}
		}
	}
}
```

## LCA

```c++
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
```

## Largest From Histogram

```c++
#include <stack>
#include <algorithm>
#include <vector>

using namespace std;

typedef long long Long;

Long calcMax(stack<pair<Long, int>>& s, int currIdx) {
    pair<Long, int> prev = s.top();
    s.pop();
    Long height = prev.first;
    int width = (s.empty() ? currIdx : currIdx - s.top().second - 1);
    return width * height;
}

Long findLargestFromHist(vector<Long>& hist) {
    int n = hist.size();
    stack<pair<Long, int>> s;
    Long result = 0;
    s.emplace(hist[0], 0);

    for (int i = 1; i < n; ++i) {
        while (!s.empty() && hist[i] < s.top().first) {
            result = max(calcMax(s, i), result);
        }
        s.emplace(hist[i], i);
    }

    while (!s.empty()) {
        result = max(calcMax(s, n), result);
    }

    return result;
}
```

## LiChaoTree

```c++
#include <bits/stdc++.h>

using namespace std;
const long long INF = 2e18;

struct Line {
    long long a, b;
    Line(): a(0), b(-INF) {}
    Line(long long a1, long long b1): a(a1), b(b1) {}
    long long f(long long x) {
        return a * x + b;
    }
};

struct Node {
    int l, r;
    long long s, e;
    Line line;
    Node(int l1, int r1, long long s1, long long e1, Line line1) : l(l1), r(r1), s(s1), e(e1), line(line1) {}
};

struct LiChaoTree {
    vector<Node> nodes;

    LiChaoTree() { nodes.emplace_back(-1, -1, -INF, INF, Line()); }
    LiChaoTree(long long s, long long e) { nodes.emplace_back(-1, -1, s, e, Line()); }

    void update(Line newLine) { update(0, newLine); }
    void update(int i, Line newLine) {
        long long s, e, mid;
        s = nodes[i].s; e = nodes[i].e;
        mid = (s + e) / 2;

        Line low, high;
        if (nodes[i].line.f(s) > newLine.f(s)) {
            low = newLine;
            high = nodes[i].line;
        } else {
            low = nodes[i].line;
            high = newLine;
        }

        if (low.f(e) < high.f(e)) {
            nodes[i].line = high;
        } else if (low.f(mid) < high.f(mid)) {
            nodes[i].line = high;
            if (nodes[i].r == -1) {
                nodes[i].r = nodes.size();
                nodes.emplace_back(-1, -1, mid + 1, e, Line());
            }
            update(nodes[i].r, low);
        } else {
            nodes[i].line = low;
            if (nodes[i].l == -1) {
                nodes[i].l = nodes.size();
                nodes.emplace_back(-1, -1, s, mid, Line());
            }
            update(nodes[i].l, high);
        }
    }

    long long query(long long x) { return query(0, x); }
    long long query(int i, long long x) {
        if (i == -1) return -INF;
        long long mid = (nodes[i].s + nodes[i].e) / 2;
        long long ret = nodes[i].line.f(x);
        if (x <= mid) return max(ret, query(nodes[i].l, x));
        else return max(ret, query(nodes[i].r, x));
    }
};

int main() {
    cin.tie(nullptr);
    ios::sync_with_stdio(false);

    LiChaoTree tree = LiChaoTree(-1e12, 1e12);

    int Q;
    cin >> Q;

    while (Q--) {
        int t;
        cin >> t;
        if (t == 1) {
            long long a, b;
            cin >> a >> b;
            tree.update(Line(a, b));
        } else {
            long long x;
            cin >> x;
            cout << tree.query(x) << '\n';
        }
    }
}
```

## LineIntersect

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long Long;
typedef pair<Long, Long> point;

int ccw(point &a, point &b, point &c) {
    point ab = {b.first - a.first, b.second - a.second};
    point bc = {c.first - b.first, c.second - b.second};
    Long ret = ab.first * bc.second - ab.second * bc.first;
    ret = -ret;
    if (ret > 0) return 1;
    else if (ret == 0) return 0;
    else return -1;
}

bool isIntersect(point a1, point b1, point a2, point b2) {
    if (a1 > b1) swap(a1, b1);
    if (a2 > b2) swap(a2, b2);
    int p = ccw(a1, b1, a2) * ccw(a1, b1, b2);
    int q = ccw(a2, b2, a1) * ccw(a2, b2, b1);
    if (p == 0 && q == 0) {
        return !(b1 < a2 || b2 < a1);
    }
    return p <= 0 && q <= 0;
}

int main() {
    point p1, p2, p3, p4;
    cin >> p1.first >> p1.second;
    cin >> p2.first >> p2.second;
    cin >> p3.first >> p3.second;
    cin >> p4.first >> p4.second;
    if (isIntersect(p1, p2, p3, p4)) {
        cout << 1 << '\n';
    } else {
        cout << 0 << '\n';
    }
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

## Merge Sort Tree

```c++
#include <algorithm>
#include <vector>

using namespace std;

struct Node {
    vector<int> subArr;
    int left, right;
    Node* leftChild = nullptr;
    Node* rightChild = nullptr;
};

void mergeSubArray(vector<int> &v1, vector<int> &v2, vector<int> &dest) {
    dest.resize(v1.size() + v2.size());
    size_t i1 = 0, i2 = 0, pos = 0;

    while (i1 < v1.size() && i2 < v2.size()) {
        if (v1[i1] <= v2[i2]) {
            dest[pos++] = v1[i1++];
        } else {
            dest[pos++] = v2[i2++];
        }
    }

    while (i1 < v1.size()) {
        dest[pos++] = v1[i1++];
    }
    while (i2 < v2.size()) {
        dest[pos++] = v2[i2++];
    }
}

Node* buildNode(int left, int right, vector<int>& arr) {
    Node *current = new Node;
    current->left = left;
    current->right = right;

    if (left == right) {
        current->subArr.push_back(arr[left]);
    } else {
        int mid = (left+right)/2;
        Node* leftChild = buildNode(left, mid, arr);
        Node* rightChild = buildNode(mid+1, right, arr);
        mergeSubArray(leftChild->subArr, rightChild->subArr, current->subArr);
        current->leftChild = leftChild;
        current->rightChild = rightChild;
    }

    return current;
}

int countBigger(Node* current, int threshold, int left, int right) {
    if (current->right < left || right < current->left) {
        return 0;
    }
    if (left <= current->left && current->right <= right) {
        auto found = upper_bound(current->subArr.begin(), current->subArr.end(), threshold);
        return current->subArr.end() - found;
    }

    return countBigger(current->leftChild, threshold, left, right)
         + countBigger(current->rightChild, threshold, left, right);
}
```

## Mo's Algorithm

```c++
#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;

struct Query {
    static int sqrtN;
    int start, end, index;
    
    bool operator<(const Query& q) const {
        if (start / sqrtN != q.start / sqrtN)
            return start / sqrtN < q.start / sqrtN;
        else return end < q.end;
    }
};
int Query::sqrtN = 0;

vector<int> mosAlg(vector<int>& arr, vector<Query>& queries) {
    // sqrt(arr의 크기)로 구간을 나누어 정렬
    sort(queries.begin(), queries.end());

    // 이 아래부터는 문제에 따라 다른 구현을 해야 함.
    // 이전에 쿼리한 구간에서 양쪽을 새 구간으로 맞추어서 결과를 구함.
    // 아래는 쿼리한 구간에서 존재하는 서로 다른 수의 개수를 구하는 예시 (BOJ 13547)
    int currCount = 0;
    vector<int> count(*max_element(arr.begin(), arr.end()) + 1);
    vector<int> answer(queries.size());
    int start = queries[0].start, end = queries[0].end;

    for (int i = start; i < end; ++i) {
        ++count[arr[i]];
        if (count[arr[i]] == 1) {
            ++currCount;
        }
    }
    answer[queries[0].index] = currCount;

    for (int i = 1; i < (int)queries.size(); ++i) {
        while (queries[i].start < start) {
            ++count[arr[--start]];
            if (count[arr[start]] == 1) {
                ++currCount;
            }
        }

        while (end < queries[i].end) {
            ++count[arr[end]];
            if (count[arr[end++]] == 1) {
                ++currCount;
            }
        }

        while (start < queries[i].start) {
            --count[arr[start]];
            if (count[arr[start++]] == 0) {
                --currCount;
            }
        }

        while (queries[i].end < end) {
            --count[arr[--end]];
            if (count[arr[end]] == 0) {
                --currCount;
            }
        }
        
        answer[queries[i].index] = currCount;
    }

    return answer;
}
```

## Persistent Segment Tree

```c++
#include <vector>
#include <algorithm>
#define MAXN 1000
#define MAXY 1000

using namespace std;

class PST {
    struct Node {
        int left, right;  // [left, right]
        int sum;
        Node *lchild, *rchild;

        Node(int left, int right) : left(left), right(right), sum(0), lchild(nullptr), rchild(nullptr) {}
    };

    Node *root[MAXN + 1];  // root[x]: tree of 0 ~ x-1
    vector<Node *> node_ptrs;

    Node *update_(Node *this_node, int y, bool is_new) {
        int left = this_node->left;
        int right = this_node->right;
        int mid = (left + right) / 2;

        Node *new_node;
        if (!is_new) {
            new_node = new Node(left, right);
            node_ptrs.push_back(new_node);
            new_node->lchild = this_node->lchild;
            new_node->rchild = this_node->rchild;
        } else {
            new_node = this_node;
        }

        // Leaf node
        if (left == right) {
            new_node->sum = this_node->sum + 1;
            return new_node;
        }

        if (y <= mid) {  // Left
            if (!new_node->lchild) {
                new_node->lchild = new Node(left, mid);
                node_ptrs.push_back(new_node->lchild);
                update_(new_node->lchild, y, true);
            } else {
                new_node->lchild = update_(new_node->lchild, y, false);
            }
        } else {  // Right
            if (!new_node->rchild) {
                new_node->rchild = new Node(mid + 1, right);
                node_ptrs.push_back(new_node->rchild);
                update_(new_node->rchild, y, true);
            } else {
                new_node->rchild = update_(new_node->rchild, y, false);
            }
        }

        int sum = 0;
        if (new_node->lchild) {
            sum += new_node->lchild->sum;
        }
        if (new_node->rchild) {
            sum += new_node->rchild->sum;
        }

        new_node->sum = sum;
        return new_node;
    }

    int get_sum_(Node *here, int b, int t) {
        if (!here || t < here->left || here->right < b) {
            return 0;
        } else if (b <= here->left && here->right <= t) {
            return here->sum;
        } else {
            return get_sum_(here->lchild, b, t) + get_sum_(here->rchild, b, t);
        }
    }

public:
    PST() {
        root[0] = new Node(0, MAXY);
        node_ptrs.push_back(root[0]);
        for (int i = 1; i <= MAXN; i++) {
            root[i] = nullptr;
        }
    }

    void update(int xi, int y) {
        if (!root[xi + 1]) {
            root[xi + 1] = update_(root[xi], y, false);
        } else {
            update_(root[xi + 1], y, true);
        }
    }

    // Sum of 0 ~ x-1
    int get_sum(int xi, int b, int t) {
        return get_sum_(root[xi + 1], b, t);
    }

    ~PST() {
        for (Node *p : node_ptrs) {
            delete p;
        }
    }
};
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

## Segment Tree with Lazy

```c++
#include <vector>
#include <cmath>

using namespace std;
using Long = long long;

template <typename T>
class SegTree {
private:
    int originCount;
    vector<T> tree, lazy;
    
    void initialize(int index, int start, int end, const vector<T>& original) {
        if (start == end) {
            tree[index] = original[start];
        } else {
            int mid = (start + end) / 2;
            initialize(index*2, start, mid, original);
            initialize(index*2+1, mid+1, end, original);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

    void propagate(int index, int start, int end) {
        if (lazy[index]) {
            tree[index] += lazy[index] * (end-start+1);
            if (start < end) {
                lazy[index*2] += lazy[index];
                lazy[index*2+1] += lazy[index];
            }
            lazy[index] = 0;
        }
    }

    T query(int index, int reqStart, int reqEnd, int treeStart, int treeEnd) {
        propagate(index, treeStart, treeEnd);

        if (reqStart <= treeStart && treeEnd <= reqEnd) {
            return tree[index];
        } else if (treeStart <= reqEnd && reqStart <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            return query(index*2, reqStart, reqEnd, treeStart, treeMed)
                 + query(index*2+1, reqStart, reqEnd, treeMed+1, treeEnd);
        } else {
            return 0;
        }
    }

    void update(T add, int index, int reqStart, int reqEnd, int treeStart, int treeEnd) {
        propagate(index, treeStart, treeEnd);

        if (reqStart <= treeStart && treeEnd <= reqEnd) {
            lazy[index] += add;
            propagate(index, treeStart, treeEnd);
        } else if (treeStart <= reqEnd && reqStart <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            update(add, index*2, reqStart, reqEnd, treeStart, treeMed);
            update(add, index*2+1, reqStart, reqEnd, treeMed+1, treeEnd);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

public:
    SegTree(const vector<T>& original) {
        originCount = (int)original.size();
        int treeHeight = (int)ceil((float)log2(originCount));
        int vecSize = (1 << (treeHeight+1));
        tree.resize(vecSize);
        lazy.resize(vecSize);
        initialize(1, 0, originCount-1, original);
    }

    T query(int start, int end) {
        return query(1, start, end, 0, originCount-1);
    }

    void update(int start, int end, T add) {
        update(add, 1, start, end, 0, originCount-1);
    }
};
```

## Segment Tree

```c++
#include <vector>
#include <cmath>

using namespace std;
using Long = long long;

template <typename T>
class SegTree {
private:
    int originCount;
    vector<T> tree;
    
    void initialize(int index, int start, int end, const vector<T>& original) {
        if (start == end) {
            tree[index] = original[start];
        } else {
            int mid = (start + end) / 2;
            initialize(index*2, start, mid, original);
            initialize(index*2+1, mid+1, end, original);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

    T query(int index, int reqStart, int reqEnd, int treeStart, int treeEnd) {
        if (reqStart <= treeStart && treeEnd <= reqEnd) {
            return tree[index];
        } else if (treeStart <= reqEnd && reqStart <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            return query(index*2, reqStart, reqEnd, treeStart, treeMed)
                 + query(index*2+1, reqStart, reqEnd, treeMed+1, treeEnd);
        } else {
            return 0;
        }
    }

    void update(T add, int index, int reqPos, int treeStart, int treeEnd) {
        if (treeStart == reqPos && treeEnd == reqPos) {
            tree[index] += add;
        } else if (treeStart <= reqPos && reqPos <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            update(add, index*2, reqPos, treeStart, treeMed);
            update(add, index*2+1, reqPos, treeMed+1, treeEnd);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

public:
    SegTree(const vector<T>& original) {
        originCount = (int)original.size();
        int treeHeight = (int)ceil((float)log2(originCount));
        int vecSize = (1 << (treeHeight+1));
        tree.resize(vecSize);
        initialize(1, 0, originCount-1, original);
    }

    T query(int start, int end) {
        return query(1, start, end, 0, originCount-1);
    }

    void update(int pos, T add) {
        update(add, 1, pos, 0, originCount-1);
    }
};
```

## SegmentTreeMaxSubarray

```c++
#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

struct Node{
    int s, m, l, r;
    Node() : s(0), m(-INF), l(-INF), r(-INF) { }
    Node operator+(Node &right) {
        Node ret;
        ret.s = s + right.s;
        ret.l = max(l, s + right.l);
        ret.r = max(right.r, r + right.s);
        ret.m = max(r + right.l, max(m, right.m));
        return ret;
    }
};

struct SegTree{
    vector<Node> arr;
    int start;
    SegTree(int n) {
        start = 1;
        while (start < n) start *= 2;
        arr.resize(start * 2);
    }

    void set(int here, int v) {
        // 0-index, require prepare()
        here += start;
        arr[here].s = v;
        arr[here].l = v;
        arr[here].r = v;
        arr[here].m = v;
    }

    void prepare() {
        for (int i = start - 1; i >= 0; i--) {
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
        }
    }

    void update(int here, int val) {
        // 0-index
        set(here, val);
        here += start;
        while (here > 0) {
            here = (here - 1) / 2;
            arr[here] = arr[here * 2] + arr[here * 2 + 1];
        }
    }

    int query(int l, int r) {
        // [l, r], 0-index
        l += start;
        r += start;
        Node retL = Node();
        Node retR = Node();
        while (l <= r) {
            if (l % 2 == 1) retL = retL + arr[l++];
            if (r % 2 == 0) retR = arr[r--] + retR;
            l /= 2, r /= 2;
        }
        return (retL + retR).m;
    }
};
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

## TernarySearch

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    int lo = 0, hi = INF;
    while (hi - lo >= 3) {
        int p = (lo * 2 + hi) / 3, q = (lo + hi * 2) / 3;
        if (f(p) <= f(q)) hi = q;
        else lo = p;
    }

    long long result = INF;
    for (int i = lo; i <= hi; i++)
        result = min(f(i), result);
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

## mulmuri's_segtree

```c++
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

#define fastio                        \
    ios_base::sync_with_stdio(false); \
    cin.tie(NULL);
#define endl '\n'

const int SIZE = 1000001;



template<typename T>
struct SegTree{
    int size, start;
    vector<T> arr;

    SegTree(int n): size(n){
        start = 1;
        while (start < size) start *= 2;
        arr.resize(start+size);
    }

    void print() {
        for (int here=start; here<start+size; here++) cout << arr[here] <<' ';
        cout << endl;
    }

    void update(int here, T val){
        here += start;
        arr[here] = val;
        while (here){
            here /= 2;
            arr[here] = merge(arr[here * 2], arr[here * 2 + 1]);
        }
    }

    T query(int l, int r){
        l += start;
        r += start;
        T ret = 0;
        while (l <= r){
            if (l % 2 == 1) ret = merge(ret, arr[l++]);
            if (r % 2 == 0) ret = merge(ret, arr[r--]);
            l /= 2; r /= 2;
        }
        return ret;
    }

    T merge(T l, T r) {
        return l + r;
    }
};



int n,m,k;
long long arr[SIZE];

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    cin >> n >> m >> k;
    SegTree<long long> st(n+1);

    for (int i=1; i<=n; i++) cin >> arr[i];
    //for (int i=1; i<=n; i++) st.update(i, arr[i]);

    for (int i=0; i<m+k; i++) {
        long long a,b,c;
        cin >> a >> b >> c;
        if (a == 1) st.update(b, c);
        else cout << st.query(b, c) << endl;
    }

}
```

## 가장 가까운 두 점

```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#define MAX 800000000

using namespace std;

struct coo {
	int x, y;
};

int square(int a);
int dist2(coo a, coo b);
bool comp_x(coo a, coo b);
bool comp_y(coo a, coo b);

// [left, right)
int find_min(vector<coo>& p, int left, int right) {
	if (left >= right - 1) {
        return MAX;
    }

	int left_min = find_min(p, left, (left+right)/2);
	int right_min = find_min(p, (left+right)/2, right);

	int min_square = min(left_min, right_min);
	double width = sqrt(min_square);

	double mid_x = (p[(left+right-1)/2].x + p[(left+right)/2].x) / 2.0;
	double left_x = mid_x - width, right_x = mid_x + width;

	vector<int> xs(right-left);
	for (int i = left; i < right; i++) {
		xs[i-left] = p[i].x;
	}

	// Find an index at which left_x < p[index].x
	int left_idx = upper_bound(xs.begin(), xs.end(), floor(left_x)) - xs.begin() + left;

	// Find an index at which p[index].x < right_x
	int right_idx = lower_bound(xs.begin(), xs.end(), ceil(right_x)) - xs.begin() + left;

	// [left_idx, right_idx)
	if (right_idx - left_idx <= 1) {
        return min_square;
    }

	vector<coo> p_in(right_idx-left_idx);
	for (int i = left_idx; i < right_idx; i++) {
		p_in[i-left_idx] = p[i];
	}
	sort(p_in.begin(), p_in.end(), comp_y);

	int center_min = MAX, bot = 0;
	for (int i = 1; i < right_idx-left_idx; i++) {
		while (square(p_in[i].y-p_in[bot].y) >= min_square && bot < i) {
            bot++;
        }
		for (int j = bot; j < i; j++) {
			center_min = min(center_min, dist2(p_in[i], p_in[j]));
		}
	}

	return min(min_square, center_min);
}

int main() {
	int n;
	cin >> n;

	vector<coo> p(n);
	for (int i = 0; i < n; i++) {
		cin >> p[i].x >> p[i].y;
    }
	sort(p.begin(), p.end(), comp_x);

	cout << find_min(p, 0, n);
}

int square(int a) {
	return a * a;
}

int dist2(coo a, coo b) {
	return square(a.x - b.x) + square(a.y - b.y);
}

bool comp_x(coo a, coo b) {
	return a.x < b.x;
}

bool comp_y(coo a, coo b) {
	return a.y < b.y;
}
```

