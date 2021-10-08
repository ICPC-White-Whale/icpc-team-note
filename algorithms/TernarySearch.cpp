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