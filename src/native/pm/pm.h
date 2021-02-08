#ifndef PM__PM_H
#define PM__PM_H

#include <vector>
//enum Status { iddle, waiting_fb_0, waiting_fb_1};

// change pm_cpp_ext.pyx accordingly
//#define PM_DTYPE float
#define PM_DTYPE double

class Drone {
 public:
  Drone();
  PM_DTYPE flight_duration();
  void reset(PM_DTYPE x0, PM_DTYPE y0, float v);
  //void get_pos(float t, float* x, float* y);
  void get_last_leg_start_pos(PM_DTYPE* x, PM_DTYPE* y);
  PM_DTYPE get_last_leg_start_time();
  void add_leg(float psi, PM_DTYPE dt);

  std::vector<float> get_psis() { return _psi;} 
  std::vector<PM_DTYPE> get_xs() {return _x;}
  std::vector<PM_DTYPE> get_ys() {return _y;}
  float _v;
  
 private:
  std::vector<PM_DTYPE> _x;
  std::vector<PM_DTYPE> _y;
  std::vector<float> _psi;
  std::vector<float> _vx;
  std::vector<float> _vy;
  std::vector<PM_DTYPE> _ts;
};


class Target {
 public:
  Target(int name, PM_DTYPE x0, PM_DTYPE y0, float v, float psi);
  int name() {return _name;};
  void get_pos(PM_DTYPE t, PM_DTYPE* x, PM_DTYPE* y);

  float _vx, _vy;
 private:
  int _name;
  PM_DTYPE _x0, _y0;
  float _v, _psi;
};


class Solver {
 public:
  Solver();
  //~Solver();
  bool init(PM_DTYPE* dp, float dv, std::vector<PM_DTYPE> tx, std::vector<PM_DTYPE> ty, std::vector<float> tv, std::vector<float> th);
  float run_sequence(std::vector<int> seq);
  float run_exhaustive(std::vector<int> &best_seq);
  void solve_1(float* psi, PM_DTYPE* dt, Target target);
  std::vector<float> get_psis() { return _drone.get_psis();} 
 private:
  std::vector<Target> _targets;
  Drone _drone;
};
#endif // PM__PM_H
