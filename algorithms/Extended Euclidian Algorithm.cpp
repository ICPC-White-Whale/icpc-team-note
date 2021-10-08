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