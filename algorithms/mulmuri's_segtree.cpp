#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

#define fastio                        \
    ios_base::sync_with_stdio(false); \
    cin.tie(NULL);
#define endl '\n'

const int SIZE = 1000001;



template<typename T>
struct SegTree{
    int size, start;
    vector<T> arr;

    SegTree(int n): size(n){
        start = 1;
        while (start < size) start *= 2;
        arr.resize(start+size);
    }

    void print() {
        for (int here=start; here<start+size; here++) cout << arr[here] <<' ';
        cout << endl;
    }

    void update(int here, T val){
        here += start;
        arr[here] = val;
        while (here){
            here /= 2;
            arr[here] = merge(arr[here * 2], arr[here * 2 + 1]);
        }
    }

    T query(int l, int r){
        l += start;
        r += start;
        T ret = 0;
        while (l <= r){
            if (l % 2 == 1) ret = merge(ret, arr[l++]);
            if (r % 2 == 0) ret = merge(ret, arr[r--]);
            l /= 2; r /= 2;
        }
        return ret;
    }

    T merge(T l, T r) {
        return l + r;
    }
};



int n,m,k;
long long arr[SIZE];

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    cin >> n >> m >> k;
    SegTree<long long> st(n+1);

    for (int i=1; i<=n; i++) cin >> arr[i];
    //for (int i=1; i<=n; i++) st.update(i, arr[i]);

    for (int i=0; i<m+k; i++) {
        long long a,b,c;
        cin >> a >> b >> c;
        if (a == 1) st.update(b, c);
        else cout << st.query(b, c) << endl;
    }

}