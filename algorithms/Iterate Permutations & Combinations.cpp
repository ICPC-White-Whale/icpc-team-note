#include <iostream>
#include <functional>

void iteratePerm(int n, int k, std::function<void(const std::vector<int>&)> callback) {
    std::vector<bool> inSelected(n+1, false);
    std::vector<int> selected;
    selected.reserve(k);
    std::function<void()> searchCases = [&]() {
        if (selected.size() == k) {
            callback(selected);
        } else {
            for (int i = 0; i < n; ++i) {
                if (!inSelected[i]) {
                    selected.push_back(i);
                    inSelected[i] = true;
                    searchCases();
                    selected.pop_back();
                    inSelected[i] = false;
                }
            }
        }
    };
    searchCases();
}

void iterateComb(int n, int k, std::function<void(const std::vector<int>&)> callback) {
    std::vector<int> selected;
    selected.reserve(k);
    std::function<void(int)> searchCases = [&](int start) {
        if (selected.size() == k) {
            callback(selected);
        } else if (n-start < k-selected.size()) {
            return;
        } else {
            selected.push_back(start);
            searchCases(start+1);
            selected.pop_back();
            searchCases(start+1);
        }
    };
    searchCases(0);
}

void iteratePermWithDup(int n, int k, std::function<void(const std::vector<int>&)> callback) {
    std::vector<int> selected;
    selected.reserve(k);
    std::function<void()> searchCases = [&]() {
        if (selected.size() == k) {
            callback(selected);
        } else {
            for (int i = 0; i < n; ++i) {
                selected.push_back(i);
                searchCases();
                selected.pop_back();
            }
        }
    };
    searchCases();
}

void iterateCombWithDup(int n, int k, std::function<void(const std::vector<int>&)> callback) {
    std::vector<int> selected;
    selected.reserve(k);
    std::function<void(int)> searchCases = [&](int start) {
        if (selected.size() == k) {
            callback(selected);
        } else if (start < n) {
            selected.push_back(start);
            searchCases(start);
            selected.pop_back();
            searchCases(start+1);
        }
    };
    searchCases(0);
}