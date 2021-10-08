#include <algorithm>
#include <vector>

using namespace std;

struct Node {
    vector<int> subArr;
    int left, right;
    Node* leftChild = nullptr;
    Node* rightChild = nullptr;
};

void mergeSubArray(vector<int> &v1, vector<int> &v2, vector<int> &dest) {
    dest.resize(v1.size() + v2.size());
    size_t i1 = 0, i2 = 0, pos = 0;

    while (i1 < v1.size() && i2 < v2.size()) {
        if (v1[i1] <= v2[i2]) {
            dest[pos++] = v1[i1++];
        } else {
            dest[pos++] = v2[i2++];
        }
    }

    while (i1 < v1.size()) {
        dest[pos++] = v1[i1++];
    }
    while (i2 < v2.size()) {
        dest[pos++] = v2[i2++];
    }
}

Node* buildNode(int left, int right, vector<int>& arr) {
    Node *current = new Node;
    current->left = left;
    current->right = right;

    if (left == right) {
        current->subArr.push_back(arr[left]);
    } else {
        int mid = (left+right)/2;
        Node* leftChild = buildNode(left, mid, arr);
        Node* rightChild = buildNode(mid+1, right, arr);
        mergeSubArray(leftChild->subArr, rightChild->subArr, current->subArr);
        current->leftChild = leftChild;
        current->rightChild = rightChild;
    }

    return current;
}

int countBigger(Node* current, int threshold, int left, int right) {
    if (current->right < left || right < current->left) {
        return 0;
    }
    if (left <= current->left && current->right <= right) {
        auto found = upper_bound(current->subArr.begin(), current->subArr.end(), threshold);
        return current->subArr.end() - found;
    }

    return countBigger(current->leftChild, threshold, left, right)
         + countBigger(current->rightChild, threshold, left, right);
}