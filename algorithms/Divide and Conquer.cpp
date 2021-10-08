// DnQ Optimization을 적용하기 위한 조건

/*
조건 1: dp[t][i] = min_{k<i} (dp[t-1][k] + C[k][i])
*/

/*
조건 2: 아래 두 조건들 중 적어도 하나를 만족

    a)  A[t][i]를 dp[t][i]를 만족시키는 최소의 k라고 할 때 아래 부등식을 만족
        A[t][i] <= A[t][i+1]
    
    b)  비용 C가 a<=b<=c<=d인 a, b, c, d에 대하여
        사각부등식 C[a][c] + C[b][d] <= C[a][d] + C[b][c] 를 만족
*/

// 위 두 조건이 만족될 경우, O(KN log N)으로 해결가능

typedef long long Long;

int L, G;
Long Ci[8001];
Long sum[8001];

Long dp[801][8001], properK[801][8001];

// 문제에 맞게 Cost 정의
Long getCost(Long a, Long b) {
    return (sum[b] - sum[a - 1]) * (b - a + 1);
}

// dp[t][i] = min_{k<i} (dp[t-1][k] + C[k][i]) 꼴의 문제를 풀고자 할 때,
// 아래 함수는 dp[t][l~r]을 채운다.
void Find(int t, int l, int r, int p, int q) {
    if (l > r) {
        return;
    }
    int mid = (l + r) >> 1;
    dp[t][mid] = -1;
    properK[t][mid] = -1;

    for (int k = p; k <= q; ++k) {
        Long current = dp[t - 1][k] + getCost(k+1, mid);
        if (dp[t][mid] == -1 || dp[t][mid] > current) {
            dp[t][mid] = current;
            properK[t][mid] = k;
        }
    }

    Find(t, l, mid - 1, p, properK[t][mid]);
    Find(t, mid + 1, r, properK[t][mid], q);
}