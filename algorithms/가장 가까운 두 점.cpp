#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#define MAX 800000000

using namespace std;

struct coo {
	int x, y;
};

int square(int a);
int dist2(coo a, coo b);
bool comp_x(coo a, coo b);
bool comp_y(coo a, coo b);

// [left, right)
int find_min(vector<coo>& p, int left, int right) {
	if (left >= right - 1) {
        return MAX;
    }

	int left_min = find_min(p, left, (left+right)/2);
	int right_min = find_min(p, (left+right)/2, right);

	int min_square = min(left_min, right_min);
	double width = sqrt(min_square);

	double mid_x = (p[(left+right-1)/2].x + p[(left+right)/2].x) / 2.0;
	double left_x = mid_x - width, right_x = mid_x + width;

	vector<int> xs(right-left);
	for (int i = left; i < right; i++) {
		xs[i-left] = p[i].x;
	}

	// Find an index at which left_x < p[index].x
	int left_idx = upper_bound(xs.begin(), xs.end(), floor(left_x)) - xs.begin() + left;

	// Find an index at which p[index].x < right_x
	int right_idx = lower_bound(xs.begin(), xs.end(), ceil(right_x)) - xs.begin() + left;

	// [left_idx, right_idx)
	if (right_idx - left_idx <= 1) {
        return min_square;
    }

	vector<coo> p_in(right_idx-left_idx);
	for (int i = left_idx; i < right_idx; i++) {
		p_in[i-left_idx] = p[i];
	}
	sort(p_in.begin(), p_in.end(), comp_y);

	int center_min = MAX, bot = 0;
	for (int i = 1; i < right_idx-left_idx; i++) {
		while (square(p_in[i].y-p_in[bot].y) >= min_square && bot < i) {
            bot++;
        }
		for (int j = bot; j < i; j++) {
			center_min = min(center_min, dist2(p_in[i], p_in[j]));
		}
	}

	return min(min_square, center_min);
}

int main() {
	int n;
	cin >> n;

	vector<coo> p(n);
	for (int i = 0; i < n; i++) {
		cin >> p[i].x >> p[i].y;
    }
	sort(p.begin(), p.end(), comp_x);

	cout << find_min(p, 0, n);
}

int square(int a) {
	return a * a;
}

int dist2(coo a, coo b) {
	return square(a.x - b.x) + square(a.y - b.y);
}

bool comp_x(coo a, coo b) {
	return a.x < b.x;
}

bool comp_y(coo a, coo b) {
	return a.y < b.y;
}