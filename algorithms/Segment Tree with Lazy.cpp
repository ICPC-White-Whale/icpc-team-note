#include <vector>
#include <cmath>

using namespace std;
using Long = long long;

template <typename T>
class SegTree {
private:
    int originCount;
    vector<T> tree, lazy;
    
    void initialize(int index, int start, int end, const vector<T>& original) {
        if (start == end) {
            tree[index] = original[start];
        } else {
            int mid = (start + end) / 2;
            initialize(index*2, start, mid, original);
            initialize(index*2+1, mid+1, end, original);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

    void propagate(int index, int start, int end) {
        if (lazy[index]) {
            tree[index] += lazy[index] * (end-start+1);
            if (start < end) {
                lazy[index*2] += lazy[index];
                lazy[index*2+1] += lazy[index];
            }
            lazy[index] = 0;
        }
    }

    T query(int index, int reqStart, int reqEnd, int treeStart, int treeEnd) {
        propagate(index, treeStart, treeEnd);

        if (reqStart <= treeStart && treeEnd <= reqEnd) {
            return tree[index];
        } else if (treeStart <= reqEnd && reqStart <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            return query(index*2, reqStart, reqEnd, treeStart, treeMed)
                 + query(index*2+1, reqStart, reqEnd, treeMed+1, treeEnd);
        } else {
            return 0;
        }
    }

    void update(T add, int index, int reqStart, int reqEnd, int treeStart, int treeEnd) {
        propagate(index, treeStart, treeEnd);

        if (reqStart <= treeStart && treeEnd <= reqEnd) {
            lazy[index] += add;
            propagate(index, treeStart, treeEnd);
        } else if (treeStart <= reqEnd && reqStart <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            update(add, index*2, reqStart, reqEnd, treeStart, treeMed);
            update(add, index*2+1, reqStart, reqEnd, treeMed+1, treeEnd);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

public:
    SegTree(const vector<T>& original) {
        originCount = (int)original.size();
        int treeHeight = (int)ceil((float)log2(originCount));
        int vecSize = (1 << (treeHeight+1));
        tree.resize(vecSize);
        lazy.resize(vecSize);
        initialize(1, 0, originCount-1, original);
    }

    T query(int start, int end) {
        return query(1, start, end, 0, originCount-1);
    }

    void update(int start, int end, T add) {
        update(add, 1, start, end, 0, originCount-1);
    }
};