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