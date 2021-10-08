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