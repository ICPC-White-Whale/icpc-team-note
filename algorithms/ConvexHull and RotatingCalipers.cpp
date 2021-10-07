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