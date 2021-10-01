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