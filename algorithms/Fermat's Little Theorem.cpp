long long pow(long long n, long long r, int MOD) {
    long long ret = 1;
    for (; r; r >>= 1) {
        if (r & 1) ret = ret * n % MOD;
        n = n * n % MOD;
    }
    return ret;
}
/* A * B^(p−2)  (mod p)
long long ans = fact[a];
ans = ans * pow(fact[b], MOD - 2, MOD) % MOD;
ans = ans * pow(fact[a - b], MOD - 2, MOD) % MOD;
*/