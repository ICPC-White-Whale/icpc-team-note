// Knuth Optimization을 적용하기 위한 조건

/*
조건 1: dp[i][j] = min_{i<k<j} (dp[i][k] + dp[k][j]) + C[i][j]
조건 2: C[a][c] + C[b][d] <= C[a][d] + C[b][c] (a <= b <= c <= d)
조건 3: C[b][c] <= C[a][d] (a <= b <= c <= d)
*/

// 위 세 조건이 만족될 경우, O(N^2)으로 해결가능

typedef long long Long;
const Long INF = 1LL<<32;

int data[1003];
Long d[1003][1003];
int p[1003][1003];

int getCost(int left, int right) {
    // define your cost function here
}

// data의 [left, right]에 값을 채우고 아래 함수를 실행하면 d에 dp값이 채워진다.
void doKnuthOpt(int left, int right) {
    for (int i = left; i <= right; i++) {
		d[i][i] = 0, p[i][i] = i;
		for (int j = i + 1; j <= right; j++) {
			d[i][j] = 0, p[i][j] = i;
        }
	}

	for (int l = 2; l <= right-left+1; l++)  {
		for (int i = left; i + l <= right; i++) {
			int j = i + l;
			d[i][j] = INF;
			for (int k = p[i][j - 1]; k <= p[i + 1][j]; k++) {
                int current = d[i][k] + d[k][j] + getCost(i, j);
				if (d[i][j] > current) {
					d[i][j] = current;
					p[i][j] = k;
				}
			}
		}
	}
}