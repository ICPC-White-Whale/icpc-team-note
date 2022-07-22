#include <vector>
#include <algorithm>
#define LEFT_MAX 1001
#define RIGHT_MAX 1001

std::vector<int> connected[LEFT_MAX];
bool visited[LEFT_MAX];
int matching[RIGHT_MAX];

bool findValidPair(int start) {
    if (visited[start]) {
        return false;
    }
    visited[start] = true;

    for (int opposite : connected[start]) {
        if (matching[opposite] == 0 || findValidPair(matching[opposite])) {
            matching[opposite] = start;
            return true;
        }
    }
    return false;
}

int bipartite(int N) {
    int result = 0;
    
    for (int i = 1; i <= N; ++i) {
        std::fill(visited, visited + N+1, false);
        result += (findValidPair(i) ? 1 : 0);
    }
    return result;
}