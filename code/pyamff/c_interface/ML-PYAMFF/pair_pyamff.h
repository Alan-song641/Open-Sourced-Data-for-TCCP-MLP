/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pyamff,PairPyAMFF);
// clang-format on
#else

#ifndef LMP_PAIR_PYAMFF_H
#define LMP_PAIR_PYAMFF_H

#include "pair.h"
#include "PyAMFFInterface.h"

namespace LAMMPS_NS {

class PairPyAMFF : public Pair {
 public:
  PairPyAMFF(class LAMMPS *);
  virtual ~PairPyAMFF();
  virtual void compute(int, int);
  virtual void init_style();
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  void transferStructureInfo();
  double** convertBox();
  // virtual void init_style();
  virtual double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);

 protected:
  PyAMFFInterface pyamff;
  int max_fps;
  double cut_global;
  double **cut;

  virtual void allocate();

};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
