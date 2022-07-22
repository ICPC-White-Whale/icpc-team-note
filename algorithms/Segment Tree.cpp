#include <vector>
#include <cmath>

using namespace std;
using Long = long long;

template <typename T>
class SegTree {
private:
    int originCount;
    vector<T> tree;
    
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

    T query(int index, int reqStart, int reqEnd, int treeStart, int treeEnd) {
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

    void update(T add, int index, int reqPos, int treeStart, int treeEnd) {
        if (treeStart == reqPos && treeEnd == reqPos) {
            tree[index] += add;
        } else if (treeStart <= reqPos && reqPos <= treeEnd) {
            int treeMed = (treeStart + treeEnd) / 2;
            update(add, index*2, reqPos, treeStart, treeMed);
            update(add, index*2+1, reqPos, treeMed+1, treeEnd);
            tree[index] = tree[index*2] + tree[index*2+1];
        }
    }

    // 0~k까지의 합산이 sum 이상이 되는 위치를 log만에 찾음.
    int findLowerBound(T sum, int index, int treeStart, int treeEnd) {
        int treeMid = (treeStart+treeEnd) / 2;
        if (treeStart == treeEnd) {
            return treeStart;
        } else if (tree[index*2] < sum) {
            return findLowerBound(sum-tree[index*2], index*2+1, treeMid+1, treeEnd);
        } else {
            return findLowerBound(sum, index*2, treeStart, treeMid);
        }
    }

public:
    SegTree(const vector<T>& original) {
        originCount = (int)original.size();
        int treeHeight = (int)ceil((float)log2(originCount));
        int vecSize = (1 << (treeHeight+1));
        tree.resize(vecSize);
        initialize(1, 0, originCount-1, original);
    }

    T query(int start, int end) {
        return query(1, start, end, 0, originCount-1);
    }

    void update(int pos, T add) {
        update(add, 1, pos, 0, originCount-1);
    }

    // 0~k까지의 합산이 sum 이상이 되는 위치를 log만에 찾음.
    int findLowerBound(T sum) {
        return findLowerBound(sum, 1, 0, originCount-1);
    }
};