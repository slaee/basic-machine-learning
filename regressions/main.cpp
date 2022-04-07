#include <iostream>
#include "../matplotlibcpp/matplotlibcpp.h"
#include "simplelinearregression.h"
using namespace std;

int main(){
    // variable Y
    vector<double> X = { 38, 50, 15, 30, 50, 38, 50, 20, 45, 50, 20, 35, 30, 43, 35, 37.5, 37, 35, 30, 45, 4, 37.5, 25, 46, 30, 200, 200, 30
    };
    
    // variable X
    vector<double> Y = { 8000, 6400, 2500, 3000, 6000, 5000, 8000, 4000, 11000, 25000, 4000, 8800, 5000, 7000, 8000, 1800, 5400, 15000, 3500, 2400, 1000, 8000, 2100, 8000, 4000, 1000, 2000, 4800
    };
    
    
    double alpha = 0.0001; // learning rate
    int epoch = 1000;// number of epochs
    SimpleLinearRegression *slr = new SimpleLinearRegression(X, Y, alpha, epoch, true);
    slr->train();
    slr->print_yhat();

    
    vector<double> Y_c = slr->predict(X);

    // denormalize Y_c
    vector<double> Y_c_denormalize;
    double Y_MAX = *max_element(Y.begin(), Y.end());
    double Y_MIN = *min_element(Y.begin(), Y.end());
    double X_MAX = *max_element(X.begin(), X.end());
    double X_MIN = *min_element(X.begin(), X.end()); 
    
    for(int i = 0; i < Y_c.size(); i++){
        Y_c_denormalize.push_back(Y_c[i] * (Y_MAX - Y_MIN) + Y_MIN);
    }

    
    double Y_c_MAX = *max_element(Y_c_denormalize.begin(), Y_c_denormalize.end()) + Y_MAX;
    double Y_c_MIN = *min_element(Y_c_denormalize.begin(), Y_c_denormalize.end()) - Y_MIN + Y_MIN;

    // Scatter plot
    matplotlibcpp::figure_size(700, 500);
    matplotlibcpp::scatter(X, Y, 25);

    double x = 40;
    double y = slr->predict(x);
    double y_denorm = y * (Y_MAX - Y_MIN) + Y_MIN;

    cout << "Prediction of " << x << " Hours Per week is " << y_denorm << " Income" << endl;
    
    matplotlibcpp::plot({X_MIN, X_MAX}, {Y_c_MIN, Y_c_MAX}, "r");
    matplotlibcpp::xlabel("Hours per Week (x)");
    matplotlibcpp::xlim(0, 80);
    matplotlibcpp::ylim(0, 35000);
    matplotlibcpp::ylabel("Income (y)");
    matplotlibcpp::title("Scatter Plot of Hours per Week and Income By Sly Kint A. Bacalso");
    matplotlibcpp::show();
};