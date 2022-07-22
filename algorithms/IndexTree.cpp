#include <bits/stdc++.h>
using namespace std;

const int SIZE = 4194304;

struct SegTree{
    int size, start;
    long long arr[SIZE];

    SegTree(int n): size(n){
        start = 1;
        while (start < size) start *= 2;
        memset(arr, 0, sizeof(arr));
    }

    void set(int here, long long val) {
        // 0-index, require prepare()
        arr[start + here] = val;
    }

    void prepare(){
        for (int i = start - 1; i; i--) {
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
        }
    }

    void update(int here, long long val){
        // 0-index
        here += start;
        arr[here] += val;
        while (here){
            here /= 2;
            arr[here] = arr[here * 2] + arr[here * 2 + 1];
        }
    }

    long long sum(int l, int r){
        // [l, r], 0-index
        l += start;
        r += start;
        long long ret = 0;
        while (l <= r){
            if (l % 2 == 1) ret += arr[l++];
            if (r % 2 == 0) ret += arr[r--];
            l /= 2; r /= 2;
        }
        return ret;
    }

    int search(int k) {
        // search kth number, k >= 1
        int pos = 1;
        while (pos < start) {
            if (k <= arr[pos * 2]) {
                pos *= 2;
            }
            else {
                k -= arr[pos * 2];
                pos = pos * 2 + 1;
            }
        }
        return pos - start;
    }
};