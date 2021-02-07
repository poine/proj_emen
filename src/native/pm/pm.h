#ifndef PM__PM_H
#define PM__PM_H

#include <vector>
//enum Status { iddle, waiting_fb_0, waiting_fb_1};

class Drone {
 public:
  Drone();
  float flight_duration();
  void reset(float x0, float y0, float v);
  //void get_pos(float t, float* x, float* y);
  void get_last_leg_start_pos(float* x, float* y);
  float get_last_leg_start_time();
  void add_leg(float psi, float dt);

  std::vector<float> get_psis() { return _psi;} 
  std::vector<float> get_xs() {return _x;}
  std::vector<float> get_ys() {return _y;}
  float _v;
  
 private:
  std::vector<float> _x;
  std::vector<float> _y;
  std::vector<float> _psi;
  std::vector<float> _vx;
  std::vector<float> _vy;
  std::vector<float> _ts;
};


class Target {
 public:
  Target(int name, float x0, float y0, float v, float psi);
  int name() {return _name;};
  void get_pos(float t, float* x, float* y);

  float _vx, _vy;
 private:
  int _name;
  float _x0, _y0, _v, _psi;
};


class Solver {
 public:
  Solver();
  //~Solver();
  bool init(float* dp, float dv, std::vector<float> tx, std::vector<float> ty, std::vector<float> tv, std::vector<float> th);
  float run(std::vector<int> seq);
  float run_exhaustive(std::vector<int> &best_seq);
  void solve_1(float* psi, float* dt, Target target);
  std::vector<float> get_psis() { return _drone.get_psis();} 
 private:
  std::vector<Target> _targets;
  Drone _drone;
};
#endif // PM__PM_H
