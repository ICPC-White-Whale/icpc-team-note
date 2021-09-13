# icpc-team-note
2021 ICPC Seoul Regional White Whale team note


# 커밋 규칙

1. 반드시 `algorithms` 폴더 아래에 각자의 소스 파일 `.cpp`을 넣는다.
   * **이름: 알고리즘 명칭**
   * **이렇게 하면 충돌이 없다!**
2. **Fork / Branch**를 해서 자신이 맡은 알고리즘을 넣는다.
3. **PR을 하고**, 다른 사람이 체크한 뒤에 Merge한다.
4. Merge를 하면 GitHub Action이 자동으로 통합된 Markdown 문서를 만들어준다.


# Code Convention

1. 중괄호 위치
   * 줄 바로 옆에
2. `using namespace std` 쓰기
3. `long long` 통일안
   1. `#define int long long` **(X)**
   2. `typedef long long ll` **(X)**
   3. `typedef long long Long` **(O)**
4. Snake Case(val_name) **vs** Camel Case(valName)
   * Camel Case
5. `pair<int, int>`를 쓸 때 `using pi = pair<int, int>`을 쓸 것인가?
   * `pair<int, int>`
6. 전역 변수를 쓰는 상황
   1. 그래프 문제 (그래프를 전역으로 선언)
   2. 함수의 파라메터로 설정할 필요가 없을 때