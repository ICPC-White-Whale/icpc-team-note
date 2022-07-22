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