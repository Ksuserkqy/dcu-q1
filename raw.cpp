// raw.cpp - Q1 (DCU)
// by Ksuserkqy(20251113620)
// Docs: https://www.ksuser.cn/dcu/
// 2025-10-14

#include <iostream>
#include <cstdlib>
#include <chrono>
using namespace std;

int main() {
    auto t1 = chrono::high_resolution_clock::now();
    
    static float A[10000][10000] = {0};
    srand(14);
    for (int i=0 ; i<10000; i++) {
        for (int j=0; j<10000; j++) {
            A[i][j] = rand() / (float)RAND_MAX;
        }
    }
    
    auto t2 = chrono::high_resolution_clock::now();
    double init_time = chrono::duration<double>(t2 - t1).count();
    cout << "初始化耗时: " << init_time << " 秒" << endl;
    auto t3 = chrono::high_resolution_clock::now();

    for (int i=0 ; i<10000; i++) {
        float sum = 0.0f;
        for (int j=0; j<10000; j++) {
            sum += A[i][j];
        }
        A[i][0] = sum;
    }
    
    auto t4 = chrono::high_resolution_clock::now();
    double sum_time = chrono::duration<double>(t4 - t3).count();
    cout << "累加计算耗时: " << sum_time << " 秒" << endl;
    
    for (int i=0; i<5; i++) {
        cout << "第 " << i << " 行的累加值 = " << A[i][0] << endl;
    }
    return 0;
}