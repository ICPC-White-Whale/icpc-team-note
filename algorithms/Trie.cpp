#include <algorithm>
#include <string>

using namespace std;

struct Trie {
    Trie* next[26];

    Trie() {
        fill(next, next + 26, nullptr);
    }

    ~Trie() {
        for (int i = 0; i < 26; i++) {
            if (next[i]) {
                delete next[i];
            }
        }
    }

    void insert(const char *key) {
        if (*key == '\0') {
            return;
        }
        int index = *key - 'a';
        if (!next[index]) {
            next[index] = new Trie;
        }
        next[index]->insert(key + 1);
    }

    void insert(const string& key) {
        insert(key.c_str());
    }
};