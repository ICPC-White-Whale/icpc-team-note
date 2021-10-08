#include <vector>
#define MAX 1001

using namespace std;

int sz[MAX];
std::vector<int> adj[MAX];

int getSz(int here,int dad) {
    sz[here] = 1;
    for (auto there : adj[here]){
        if (there == dad) {
            continue;
        }
        sz[here]+=getSz(there,here);
    }
    return sz[here];
}
 
int get_centroid(int here, int dad, int cap) {
    //cap <---- (tree size)/2
    for(auto there : adj[here]){
        if (there == dad) {
            continue;
        }
        if(sz[there] > cap) {
            return get_centroid(there,here,cap);
        }
    }
    return here;
}

int main() {
    int root = 1;
    getSz(root, -1);
    int center = get_centroid(1, -1, sz[root]/2);
    return 0;
}