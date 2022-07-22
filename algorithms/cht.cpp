#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

#define SIZE 100001



template<typename T>
struct CHT {
	vector< pair<T, T> > stk;
	long long idx;

    CHT() : idx(0) {
		idx = 0; stk.clear();
	}

    pair<double,double> cross(long long a, long long b){
		double t1 = (double)(stk[b].second - stk[a].second);
		double t2 = (double)(stk[a].first - stk[b].first);
		return {t1,t2};
	}

    bool comp(pair<double, double> a, pair<double, double> b) {
        return a.first/a.second > b.first/b.second;
    }

	void insert(T a, T b){
		stk.emplace_back(a, b);
		while(stk.size() > 2 &&
		comp( cross(stk.size()-3, stk.size()-2) , cross(stk.size()-2, stk.size()-1)) ) {
			stk[stk.size()-2] = stk.back();
			stk.pop_back();
		}
	}

    pair<T,T> query(T x) {
        if (stk.size() == 0) return {0,0};

        int l = 0, r = stk.size() - 1;
		while(l < r) {
			int m = (l + r)/2;
			if (comp( cross(m,m+1), {x,1}) ) r = m;
			else l = m + 1;
		}
        return {stk[l].first, stk[l].second};
    }

};





int n;

long long pl;
long long l[SIZE];
long long p[SIZE];
long long s[SIZE];
long long dp[SIZE];

int main() {
    cin >> n;

    for (int i=n-1; i>0; i--) cin >> l[i];
    for (int i=1; i<=n; i++) l[i] += l[i-1];

    for (int i=n-1; i>0; i--) cin >> p[i] >> s[i];

    CHT<long long> cht;


    for (int i=0; i<n; i++) {
        auto[bj, cj] = cht.query(s[i]);
        dp[i] = s[i] * bj + cj + s[i]*l[i] + p[i]; // dp(i) = a(i) * b(j) + c(j) + d(i)
        cht.insert(-l[i], dp[i]);
    };

    cout << dp[n-1];
}