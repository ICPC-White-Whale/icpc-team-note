#include <vector>
#include <algorithm>
#define MAXN 1000
#define MAXY 1000

using namespace std;

class PST {
    struct Node {
        int left, right;  // [left, right]
        int sum;
        Node *lchild, *rchild;

        Node(int left, int right) : left(left), right(right), sum(0), lchild(nullptr), rchild(nullptr) {}
    };

    Node *root[MAXN + 1];  // root[x]: tree of 0 ~ x-1
    vector<Node *> node_ptrs;

    Node *update_(Node *this_node, int y, bool is_new) {
        int left = this_node->left;
        int right = this_node->right;
        int mid = (left + right) / 2;

        Node *new_node;
        if (!is_new) {
            new_node = new Node(left, right);
            node_ptrs.push_back(new_node);
            new_node->lchild = this_node->lchild;
            new_node->rchild = this_node->rchild;
        } else {
            new_node = this_node;
        }

        // Leaf node
        if (left == right) {
            new_node->sum = this_node->sum + 1;
            return new_node;
        }

        if (y <= mid) {  // Left
            if (!new_node->lchild) {
                new_node->lchild = new Node(left, mid);
                node_ptrs.push_back(new_node->lchild);
                update_(new_node->lchild, y, true);
            } else {
                new_node->lchild = update_(new_node->lchild, y, false);
            }
        } else {  // Right
            if (!new_node->rchild) {
                new_node->rchild = new Node(mid + 1, right);
                node_ptrs.push_back(new_node->rchild);
                update_(new_node->rchild, y, true);
            } else {
                new_node->rchild = update_(new_node->rchild, y, false);
            }
        }

        int sum = 0;
        if (new_node->lchild) {
            sum += new_node->lchild->sum;
        }
        if (new_node->rchild) {
            sum += new_node->rchild->sum;
        }

        new_node->sum = sum;
        return new_node;
    }

    int get_sum_(Node *here, int b, int t) {
        if (!here || t < here->left || here->right < b) {
            return 0;
        } else if (b <= here->left && here->right <= t) {
            return here->sum;
        } else {
            return get_sum_(here->lchild, b, t) + get_sum_(here->rchild, b, t);
        }
    }

public:
    PST() {
        root[0] = new Node(0, MAXY);
        node_ptrs.push_back(root[0]);
        for (int i = 1; i <= MAXN; i++) {
            root[i] = nullptr;
        }
    }

    void update(int xi, int y) {
        if (!root[xi + 1]) {
            root[xi + 1] = update_(root[xi], y, false);
        } else {
            update_(root[xi + 1], y, true);
        }
    }

    // Sum of 0 ~ x-1
    int get_sum(int xi, int b, int t) {
        return get_sum_(root[xi + 1], b, t);
    }

    ~PST() {
        for (Node *p : node_ptrs) {
            delete p;
        }
    }
};