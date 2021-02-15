#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>

#include "pm/pm.h"

//
// Drone
//
Drone::Drone() {}


void Drone::reset(PmType x0, PmType y0, float v) {
  _v = v;
  _x.clear(); _y.clear(); _psi.clear(); _vx.clear(); _vy.clear(); _ts.clear();
  _x.push_back(x0); _y.push_back(y0); _ts.push_back(0.);
}

void Drone::get_last_leg_start_pos(PmType* x, PmType* y) {
  *x = _x.back(); *y = _y.back();
}

//PmType Drone::get_last_leg_start_time() { return _ts.back(); }
PmType Drone::flight_duration() { return _ts.back(); }

void Drone::add_leg(float psi, PmType dt) {
  _ts.push_back(_ts.back()+dt);
  _psi.push_back(psi);
  float vx=_v*cos(psi), vy=_v*sin(psi);
  _vx.push_back(vx); _vy.push_back(vy);
  PmType x=_x.back()+vx*dt, y=_y.back()+vy*dt;
  _x.push_back(x); _y.push_back(y);
}

//
// Target
//
Target::Target(int name, PmType x0, PmType y0, float v, float psi):
  _name(name), _x0(x0), _y0(y0), _v(v), _psi(psi){
  _vx = _v*cos(psi);
  _vy = _v*sin(psi);
}

void Target::get_pos(PmType t, PmType* x, PmType* y) {
  *x = _x0 + _vx*t;
  *y = _y0 + _vy*t;
}


//
// Solver
//
Solver::Solver() {}

bool Solver::init(PmType* dp, float dv, std::vector<PmType> tx, std::vector<PmType> ty, std::vector<float> tv, std::vector<float> th) {
  _drone.reset(dp[0], dp[1], dv);
  for (unsigned int i=0; i<tv.size(); i++) {
    Target t = Target(i, tx[i], ty[i], tv[i], th[i]);
    _targets.push_back(t);
  }
  return true;
}


PmType Solver::search_sa(std::vector<int> start_seq, unsigned int epochs, float T0) {
}



void solve_quadratic(PmType a, PmType b, PmType c, PmType* l0, PmType*l1) {
  PmType delta = b*b-4*a*c;
  if (delta >= 0.) {
    *l0 = (-b + sqrt(delta))/2./a;
    *l1 = (-b - sqrt(delta))/2./a;
  }
  else {
    std::printf(" complex roots\n");
    std::printf(" %Lf %Lf %Lf\n", a, b, c);
  }
}

void delta_v(float dv, float dpsi, float tvx, float tvy, float*dvx, float* dvy) {
  *dvx = dv*cos(dpsi)-tvx; *dvy = dv*sin(dpsi)-tvy; 
}
PmType _norm(PmType vx, PmType vy) { return sqrt(vx*vx+vy*vy); }
PmType _scal_prod(PmType ax, PmType ay, PmType bx, PmType by) { return ax*bx+ay*by; }
unsigned long int _fact(unsigned long int n) {return (n == 1 || n == 0) ? 1 : _fact(n - 1) * n; }

void Solver::solve_1(float* psi, PmType* dt, Target target) {
  PmType dx, dy, tx, ty;
  _drone.get_last_leg_start_pos(&dx, &dy);
  target.get_pos(_drone.flight_duration(), &tx, &ty);
  PmType dpx = dx-tx, dpy = dy-ty;
  PmType dn = _norm(dpx, dpy);
  if (dn<1e-12) {*dt=0.; *psi=0.; return;} // already over the target, don't bother
  PmType a = dpy*_drone._v, b = -dpx*_drone._v;
  PmType c = dpy*target._vx-dpx*target._vy;
  float dvx, dvy;
  if (abs(a)<1e-12) {
    float psi1 = asin(c/b);
    *psi = psi1;
    delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    if (_scal_prod(dvx, dvy, dpx, dpy) > 0) {
      *psi = M_PI - *psi;
      delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    }
  }
  else { // Yay!!! psis = 2*np.arctan(np.roots([a+c, -2*b, c-a]))
    PmType l1,l2;
    solve_quadratic(a+c, -2*b, c-a, &l1, &l2);
    *psi = 2*atan(l1);
    delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    if (_scal_prod(dvx, dvy, dpx, dpy) > 0) {
      *psi = 2*atan(l2);
      delta_v(_drone._v, *psi, target._vx, target._vy, &dvx, &dvy);
    }
  }
  *dt = dn/_norm(dvx, dvy);
}


PmType Solver::run_sequence(std::vector<int> seq) {
  _drone.reset(_drone.get_xs().front(), _drone.get_ys().front(), _drone._v);
  for (int tid:seq) {
    float psi; PmType dt;
    solve_1(&psi, &dt, _targets[tid]);
    //if (isnan(dt)) { std::printf("##### GOTCHA");}
    _drone.add_leg(psi, dt);
    //if (_drone.flight_duration() > 1e16) return _drone.flight_duration(); // abort
  }
  //std::printf("C dur %Le\n", _drone.flight_duration());
  return _drone.flight_duration();
}


PmType Solver::run_sequence_threshold(std::vector<int> seq, PmType max_t) {
  _drone.reset(_drone.get_xs().front(), _drone.get_ys().front(), _drone._v);
  for (int tid:seq) {
    float psi; PmType dt;
    solve_1(&psi, &dt, _targets[tid]);
    _drone.add_leg(psi, dt);
    if (_drone.flight_duration() >= max_t) break;
  }
  return _drone.flight_duration();
}



PmType Solver::run_exhaustive(std::vector<int> &best_seq) {
  std::vector<int> seq;
  for (unsigned int i=0; i<_targets.size(); i++)
    seq.push_back(i);
  PmType best_cost = std::numeric_limits<PmType>::infinity();
  std::vector<int> _best_seq;
  do {
      PmType cost = run_sequence(seq);
      if (cost < best_cost) {
	best_cost = cost;
	_best_seq = std::vector<int>(seq);
      }
    } while(std::next_permutation(seq.begin(), seq.end())); 
  for (int _s:_best_seq)
    best_seq.push_back(_s);
  return best_cost;
}


PmType Solver::run_random(std::vector<int> &seq) {
  for (unsigned int i=0; i<_targets.size(); i++)
    seq.push_back(i);
  std::random_shuffle( seq.begin(), seq.end() );
  return run_sequence(seq);
}
