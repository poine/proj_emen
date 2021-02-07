#include <iostream>
#include <math.h>
#include <algorithm>
//#include <cmath>
#include <string>
//#include <boost/algorithm/string/join.hpp>

#include "pm/pm.h"

Drone::Drone() {
}

float Drone::flight_duration() { return _ts.back(); }

void Drone::reset(float x0, float y0, float v) {
  _v = v;
  _x.clear(); _y.clear(); _psi.clear(); _vx.clear(); _vy.clear(); _ts.clear();
  _x.push_back(x0); _y.push_back(y0); _ts.push_back(0.);
}

// void Drone::get_pos(float t, float* x, float* y) {
//   int _i=0; float dt = t-_ts[_i];
//   *x = _x[_i] + _vx[_i]*dt;
//   *y = _y[_i] + _vy[_i]*dt;
// }

void Drone::get_last_leg_start_pos(float* x, float* y) {
  *x = _x.back(); *y = _y.back();
}

float Drone::get_last_leg_start_time() {
  return _ts.back();
}

void Drone::add_leg(float psi, float dt) {
  _ts.push_back(_ts.back()+dt);
  _psi.push_back(psi);
  float vx=_v*cos(psi), vy=_v*sin(psi);
  _vx.push_back(vx); _vy.push_back(vy);
  float x=_x.back()+vx*dt, y=_y.back()+vy*dt;
  _x.push_back(x); _y.push_back(y);
}

Target::Target(int name, float x0, float y0, float v, float psi):
  _name(name), _x0(x0), _y0(y0), _v(v), _psi(psi){
  _vx = _v*cos(psi);
  _vy = _v*sin(psi);
}

void Target::get_pos(float t, float* x, float* y) {
  *x = _x0 + _vx*t;
  *y = _y0 + _vy*t;
}


Solver::Solver() {
  //std::printf("Solver::Solver()\n");
}

bool Solver::init(float* dp, float dv, std::vector<float> tx, std::vector<float> ty, std::vector<float> tv, std::vector<float> th) {
  //std::printf("initialized\n");
  //std::printf("  drone (%f %f) %f\n", dp[0], dp[1], dv);
  _drone.reset(dp[0], dp[1], dv);
  // for (float v:tv) {
  //   std::printf("  v %f\n", v);
  // }
  for (unsigned int i=0; i<tv.size(); i++) {
    Target t = Target(i, tx[i], ty[i], tv[i], th[i]);
    _targets.push_back(t);
  }
  return true;
}

void solve_quadratic(float a, float b, float c, float* l0, float*l1) {
  float delta = b*b-4*a*c;
  //std::printf(" delta %f\n", delta);
  if (delta >= 0.) {
    *l0 = (-b + sqrt(delta))/2./a;
    *l1 = (-b - sqrt(delta))/2./a;
  }
  else {
    std::printf(" complex roots\n");
  }
    
}

void delta_v(float dv, float dpsi, float tvx, float tvy, float*dvx, float* dvy) {
  *dvx = dv*cos(dpsi)-tvx; *dvy = dv*sin(dpsi)-tvy; 
}
float _norm(float vx, float vy) { return sqrt(vx*vx+vy*vy); }
float _scal_prod(float ax, float ay, float bx, float by) { return ax*bx+ay*by; }
unsigned long int _fact(unsigned long int n) {return (n == 1 || n == 0) ? 1 : _fact(n - 1) * n; }

void Solver::solve_1(float* psi, float* dt, Target target) {
  //std::printf("  solve1\n");
  float dx, dy, tx, ty;
  _drone.get_last_leg_start_pos(&dx, &dy);
  target.get_pos(_drone.get_last_leg_start_time(), &tx, &ty);
  //std::printf(" target %d pos (%f %f) drone (%f %f)\n", target.name(), tx, ty, dx, dy);
  float dpx = dx-tx, dpy = dy-ty;
  float dn = _norm(dpx, dpy);
  if (dn<1e-12) {*dt=0.; *psi=0.; return;} // already there
  float a = dpy*_drone._v, b = -dpx*_drone._v;
  float c = dpy*target._vx-dpx*target._vy;
  if (abs(a)<1e-12) {
    //std::printf("FIXME#####\n");
    float psi1 = asin(c/b);
    float psi2 = M_PI - psi1;
    float dvx, dvy;
    *psi = psi1;
    delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    if (_scal_prod(dvx, dvy, dpx, dpy) > 0) {
      *psi = psi2;
      delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    }
    *dt = dn/_norm(dvx, dvy);
  }
  else {
    // Yay!!! psis = 2*np.arctan(np.roots([a+c, -2*b, c-a]))
    float l1,l2;
    solve_quadratic(a+c, -2*b, c-a, &l1, &l2);
    float psi1=2*atan(l1), psi2=2*atan(l2);
    //std::printf("psis: %f %f\n", psi1, psi2);
    float dvx, dvy;
    *psi = psi1;
    delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    if (_scal_prod(dvx, dvy, dpx, dpy) > 0) {
      *psi = psi2;
      delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    }
    *dt = dn/_norm(dvx, dvy);
  }
}

float Solver::run(std::vector<int> seq) {
  //std::printf("  run\n");
  _drone.reset(_drone.get_xs().front(), _drone.get_ys().front(), _drone._v);
  for (int tid:seq) {
    float psi, dt;
    solve_1(&psi, &dt, _targets[tid]);
    _drone.add_leg(psi, dt);
    //std::printf("psi dt %f %f\n", psi, dt);
  }
  //std::printf("dur: %f\n", _drone.flight_duration());
  return _drone.flight_duration();
}


float Solver::run_exhaustive(std::vector<int> &best_seq) {
  std::vector<int> seq;
  for (unsigned int i=0; i<_targets.size(); i++)
    seq.push_back(i);
  float best_cost = std::numeric_limits<float>::infinity();
  std::vector<int> _best_seq;
  do {
      float cost = run(seq);
      if (cost < best_cost) {
	best_cost = cost;
	_best_seq = std::vector<int>(seq);
      }
    } while(std::next_permutation(seq.begin(), seq.end())); 
  for (int _s:_best_seq)
    best_seq.push_back(_s);
  return best_cost;
}
