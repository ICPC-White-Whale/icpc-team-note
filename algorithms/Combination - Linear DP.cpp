#include <vector>

using Long = long long;

const Long MAX = 400001;
const Long MOD = 1000000007;

Long factorial(Long n) {
    static std::vector<Long> dp(MAX+1, -1);
    if (n == 0) {
        return 1;
    } else if (dp[n] != -1) {
        return dp[n];
    }
    return dp[n] = (factorial(n-1) * n) % MOD;
}

Long getModPow(Long val, Long exp) {
    if (val == 0) {
        return 0;
    } else if (exp == 0) {
        return 1;
    } else {
        Long pow = getModPow(val, exp/2);
        return (((pow * pow) % MOD) * (exp%2 == 0 ? 1 : val)) % MOD;
    }
}

Long getModInverse(Long val) {
    return getModPow(val, MOD-2);
}

Long getComb(Long n, Long k) {
    if (n < k) {
        return 0;
    }
    return (factorial(n) * getModInverse((factorial(n-k)*factorial(k)) % MOD)) % MOD;
}

Long getCatalan(Long n) {
    return (getComb(2*n, n) * getModInverse(n+1)) % MOD;
}