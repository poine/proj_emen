#ifndef PM__PM_H
#define PM__PM_H

#include <vector>

// change pm_cpp_ext.pyx accordingly
typedef long double PmType; // fails at 15640
//typedef double PmType; // fails at 480
//typedef float PmType; // fails scen1 from 120 targets

class Drone {
 public:
  Drone();
  PmType flight_duration();
  PmType get_last_leg_start_time();
  void reset(PmType x0, PmType y0, float v);
  void get_last_leg_start_pos(PmType* x, PmType* y);
  void add_leg(float psi, PmType dt);

  std::vector<float> get_psis() { return _psi;} 
  std::vector<PmType> get_xs() {return _x;}
  std::vector<PmType> get_ys() {return _y;}
  float _v;
  
 private:
  std::vector<PmType> _x;
  std::vector<PmType> _y;
  std::vector<float> _psi;
  std::vector<float> _vx;
  std::vector<float> _vy;
  std::vector<PmType> _ts;
};


class Target {
 public:
  Target(int name, PmType x0, PmType y0, float v, float psi);
  int name() {return _name;};
  void get_pos(PmType t, PmType* x, PmType* y);

  float _vx, _vy;
 private:
  int _name;
  PmType _x0, _y0;
  float _v, _psi;
};


class Solver {
 public:
  Solver();
  //~Solver();
  bool init(PmType* dp, float dv, std::vector<PmType> tx, std::vector<PmType> ty, std::vector<float> tv, std::vector<float> th);
  PmType search_sa(std::vector<int> start_seq, unsigned int nepoch, float T0, std::vector<int> &best_seq, int display);
  PmType search_exhaustive(std::vector<int> &best_seq);
  PmType run_sequence(std::vector<int> seq);
  PmType run_sequence_threshold(std::vector<int> seq, PmType max_t);
  PmType run_sequence_random(std::vector<int> &seq);
  void solve_1(float* psi, PmType* dt, Target target);
  std::vector<float> get_psis() { return _drone.get_psis();} 
 private:
  std::vector<Target> _targets;
  Drone _drone;
};
#endif // PM__PM_H
