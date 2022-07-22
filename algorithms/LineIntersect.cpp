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