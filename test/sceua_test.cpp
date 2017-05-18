#include "test_pch.h"
#include "core/sceua_optimizer.h"

using namespace std;
using namespace shyft::core::optimizer;

struct x2_fx:public ifx {
    size_t n_eval=0;
    double evaluate(const vector<double>& xv) {
        double y=0;
        for(auto x:xv )
            y += x*x;
        n_eval++;
        return y;
    }
};

struct fx_complex:public ifx {
    size_t n_eval=0;
    double evaluate(const vector<double>& xv) {
        double r2=0.0;
        for(auto x:xv) r2+= x*x;
        double r=sqrt(r2);
        ++n_eval;
        double y= 1.0+pow(r/3.1415,3) + cos(r);/// range 0.702.. +oo, minimum r=2.5,
        return y;
    }
};
TEST_SUITE("sceua") {
TEST_CASE("test_basic") {
    sceua opt;
    const size_t n=2;
    double x[2]={-1.0, 2.5};
    double x_min[2]= {-10,-10};
    double x_max[2]= { 10, 10.0};
    const double eps=1e-6;
    double x_eps[2]= {eps,eps};
    double y=-1;
    x2_fx f_basic;
    const size_t max_iterations=20000;
    auto r=opt.find_min(n,x_min,x_max,x,y,f_basic,eps, -1,-2,x_eps,max_iterations);
    TS_ASSERT_LESS_THAN(f_basic.n_eval, max_iterations);
    TS_ASSERT_EQUALS(r,OptimizerState::FinishedXconvergence);
    TS_ASSERT_DELTA(y,0.0,1e-5);
    TS_ASSERT_DELTA(x[0],0.0,1e-5);
    TS_ASSERT_DELTA(x[1],0.0,1e-5);
    if(getenv("SHYFT_VERBOSE")) {
        cout<<endl<<"Found solution x{"<<x[0]<<","<<x[1]<<"} -> "<<y<<endl<<"\t n_iterations:"<<f_basic.n_eval<<endl;
    }
}

TEST_CASE("test_complex") {
    sceua opt;
    const size_t n=2;
    double x[2]={-4.01, -7.5};
    double x_min[2]= {-10,-10};
    double x_max[2]= { 10, 10.0};
    const double eps=1e-5;
    double x_eps[2]= {eps,eps};
    double y=-1;
    double y_eps=1e-3;
    fx_complex f_complex;
    const size_t max_iterations=150000;
    auto r=opt.find_min(n,x_min,x_max,x,y,f_complex,y_eps, -1,-2,x_eps,max_iterations);
	TS_ASSERT_LESS_THAN(f_complex.n_eval, max_iterations + 1000);
    TS_ASSERT_EQUALS(r,OptimizerState::FinishedFxConvergence);
    TS_ASSERT_DELTA(y,0.7028,1e-3);
    double rr=sqrt(x[0]*x[0]+x[1]*x[1]);
    TS_ASSERT_DELTA(rr,2.5,1e-2);
    if(getenv("SHYFT_VERBOSE")) {
        cout<<endl<<"Found solution x{"<<x[0]<<","<<x[1]<<"}(r="<<rr <<") -> "<<y<<endl<<"\t n_iterations:"<<f_complex.n_eval<<endl;
    }
    f_complex.n_eval=0;
    x[0]=-4.01;
    x[1]=-7.5;
    r=opt.find_min(n,x_min,x_max,x,y,f_complex,eps, -1,-2,x_eps,100);
    TS_ASSERT_EQUALS(r,OptimizerState::FinishedMaxIterations);
    if(getenv("SHYFT_VERBOSE")) {
        cout<<endl<<"2. approximate Found solution x{"<<x[0]<<","<<x[1]<<"}(r="<<rr <<") -> "<<y<<endl<<"\t n_iterations:"<<f_complex.n_eval<<endl;
    }
}
}
