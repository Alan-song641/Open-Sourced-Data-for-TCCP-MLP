#include "pair_pyamff.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "domain.h" //I added this -> SOMETHING STUPID
#include "lattice.h" //I added this -> SOMETHING STUPID
#include <stdio.h> //I added this -> SOMETHING STUPID

#include <cmath>
#include <cstring>
#include <iostream>
#include <ostream>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;



PairPyAMFF::PairPyAMFF(LAMMPS *lmp) : Pair(lmp) 
{
    respa_enable = 0;
    writedata = 1;
    // pyamff = new PyAMFF();

}

PairPyAMFF::~PairPyAMFF()
{
    // ~pyamff;
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

void PairPyAMFF::allocate() {

    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag,n+1,n+1,"pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++)
        setflag[i][j] = 0;

    memory->create(cutsq,n+1,n+1,"pair:cutsq"); //
    memory->create(cut,n+1,n+1,"pair:cut"); 
}

//may need to do some initialize here, not sure what's going to happen in parallel
void PairPyAMFF::init_style() {
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;
  neighbor->requests[irequest]->half = 0;
}

void PairPyAMFF::compute(int eflag, int vflag) {

  //   double evdwl = 0.0;
  // std::vector<double> R_all;
  //handle LAMMPS flags
  //setup (nnp) or init(morse/ts)
  //ev_init(eflag,vflag)
  //see if I can see what run so we can reset
  if(eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
  transferStructureInfo();
  // Calculate forces of local and ghost atoms.
  double energy = pyamff.getForcesEnergy(atom->f);
  // Add energy contribution to total energy.
  if (eflag_global)
     ev_tally(0,0,atom->nlocal,1,energy,0.0,0.0,0.0,0.0,0.0);
 // If virial needed calculate via F dot r.
  if (vflag_fdotr) virial_fdotr_compute();

}

void PairPyAMFF::transferStructureInfo() {
  int i, j, jnum;
  int *jlist;
  int total_atoms = atom->nlocal + atom->nghost;

      //set maxneighs and total atoms
  pyamff.setMaxNeighs(neighbor->oneatom);
  pyamff.updateTotalAtoms((long)(atom->nlocal+atom->nghost));
  //update positions
  for(int k=0; k < total_atoms; k++) {
    pyamff.updatePosition(k,atom->tag[k],atom->x[k]);
  }
  
  //getNeighborList and Distances
  //loop over central atoms
  for(int ii=0; ii < list->inum+list->gnum; ii++) {//number of atoms with neighbor lists, need to include ghost atoms in neighbor 
    i = list->ilist[ii]; //get local atom index of central atom from local indices of atoms
    jnum = list->numneigh[i]; //get number of neighbors for i from number of neighbors of each atom
    jlist = list->firstneigh[i]; //create array of neighbors of atom i from local index of neighboring atom
    //get positions
    //loop over neighbors
    cout << "atom " << i << " has " << jnum << "atoms:" << endl;
    for(int jj=0; jj<jnum; jj++) {
      j = jlist[jj]; //extract local atom index of neighbor jj
      j &= NEIGHMASK; //this masks if repeated or not, no its' for parallel
      //will not work if atoms has original negative position
      // if (!(atom->x[j][1] < 0 || atom->x[j][2] < 0 || atom->x[j][0] < 0)) {
        pyamff.addNeighbor(i,j);//tag tells you which atom it mirrors
      cout << "\t" << j << ": (" << atom->x[j][0] << "," << atom->x[j][1] << "," << atom->x[j][2] << ")" << endl;
      if(jj < jnum-1) {
        cout << ",";
      }else{
        cout << endl;
      }
    }
  }
}


void PairPyAMFF::settings(int narg, char ** arg) {
    //find better way for this.
          // string s("10.0");
          // char in_cut[5];
          // strcpy(in_cut,s.c_str());
    string ten="10.0",hundred="100";
    const char *in_cut = ten.c_str(), *in_max=hundred.c_str();
    switch (narg)
    {
      case 0:
          cut_global = utils::numeric(FLERR,in_cut, false, lmp);
          max_fps = utils::numeric(FLERR,in_max, false, lmp);
          break;
      case 1:
          cut_global = utils::numeric(FLERR,arg[0], false, lmp);
          max_fps = utils::numeric(FLERR,in_max, false, lmp);
          break;
      case 2:
          cut_global = utils::numeric(FLERR,arg[0], false, lmp);
          max_fps = utils::numeric(FLERR,arg[1], false, lmp);
          break;
    default:
          error->all(FLERR, "Illegal pair_style command");
          break;
    }
    // reset cutoffs that have been explicitly set

    if (allocated) {
        int i,j;
        for (i = 1; i <= atom->ntypes; i++)
        for (j = i; j <= atom->ntypes; j++) {
            if (setflag[i][j]) cut[i][j] = cut_global;
            if (setflag[j][i]) cut[j][i] = cut_global;
        }
    }

    if(domain->box_exist) {
      double box[3][3] = {//use sub so when in parallel, get subdomain. doesn't matter in serial
          {domain->subhi[0]-domain->sublo[0],0,0},
          {0,domain->subhi[1]-domain->sublo[1],0},
          {0,0,domain->subhi[2]-domain->sublo[2]}
        };
      if(domain->triclinic) {
          box[1][0] = domain->xy;
          box[2][1] = domain->xz;
          box[2][2] = domain->yz;
      }
    double bounds[3][2] = {
      {domain->sublo[0], domain->subhi[0]},
      {domain->sublo[1], domain->subhi[1]},
      {domain->sublo[2], domain->subhi[2]},
    };
    pyamff.initialize(max_fps,(long) atom->natoms, (long) atom->ntypes,atom->type, atom->mass, atom->x, box, bounds, !domain->triclinic);

  }else {
    error->all(FLERR, "ERROR: THERE IS NO BOX! PYAMFF REQUIRES A BOX/CELL.");
  }
}

void PairPyAMFF::coeff(int narg, char **arg) {
    //this number should actually match fpParas
    if (narg < 2 || narg > 3)
        error->all(FLERR,"Incorrect args for pair coefficients");
    if(!allocated) {
        allocate();
    }
    int ilo,ihi,jlo,jhi;
    utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
    utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

    double cut_one = cut_global;
    if (narg == 3) cut_one = utils::numeric(FLERR,arg[2],false,lmp);
    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo,i); j <= jhi; j++) {
        cut[i][j] = cut_one;
        setflag[i][j] = 1;
        count++;
        }
    }
    if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

double PairPyAMFF::init_one(int i,int j){ 
    return cut[i][j];
}
void PairPyAMFF::write_restart(FILE *f){}
void PairPyAMFF::read_restart(FILE *f){
  read_restart_settings(f);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,f,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,f,nullptr,error);
        }
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}
void PairPyAMFF::write_restart_settings(FILE *f){
  write_restart_settings(f);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,f);
      if (setflag[i][j]) {
        fwrite(&cut[i][j],sizeof(double),1,f);
      }
    }
}

void PairPyAMFF::read_restart_settings(FILE *f){
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,f,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,f,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,f,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}