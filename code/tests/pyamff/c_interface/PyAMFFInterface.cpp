//-----------------------------------------------------------------------------------
// eOn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// A copy of the GNU General Public License is available at
// http://www.gnu.org/licenses/
//-----------------------------------------------------------------------------------

#include "PyAMFFInterface.h"
#include "math.h"
#include <set>
#include <ostream>
#include <iostream>
#include <vector>
#include <locale>
#include <map>
#include <algorithm>


using namespace std;

PyAMFFInterface::PyAMFFInterface(void)
{
    new_pyamff = true; 
    return;
}

PyAMFFInterface::~PyAMFFInterface(){
    cleanMemory();
}

void PyAMFFInterface::cleanMemory(void)
{
    if(new_pyamff != true ){
        new_pyamff = true;
        //also call NN cleanup
        cleanup();
        nncleanup();
    }
    return;
}

void PyAMFFInterface::updateTotalAtoms(long tatoms) {
    this->total_atoms = tatoms;
    this->pool_positions = new double[tatoms*3];
    this->neighs.resize(tatoms);
    this->adjust.resize(tatoms);
    setMaxNeighs(this->max_neighs);
    this->num_neighs = new int[tatoms];
    this->neigh_tag = new int[tatoms];
    for(int i=0; i<tatoms; i++) {
        this->num_neighs[i] = 0;
    }
}
//decrease the number of arguments
void PyAMFFInterface::initialize(int max_fps, long natoms, int ntypes, int* types, double *mass, double** pos_car, double dbox[3][3], double bounds[3][2], bool ortho) {
    //need to check if passed cutoff is okay for what pyamff sees
    this->max_fps=max_fps;
    this->num_atoms = natoms;
    this->total_atoms = natoms;
    this->num_types = ntypes;
    this->atomicNrs = new int[natoms];
    this->unique = new int[ntypes];
    this->positions = new double[natoms*3];
    this->pool_positions = new double[natoms*3];
    this->forces = new double[natoms*3];
    this->noNeighs = true;
    this->neighs.resize(natoms);
    this->adjust.resize(natoms);
    this->num_neighs = new int[natoms];
    vector<int> unique_atomicNrs;
    // HERE HERE
    //setup neighborlist and distances, num_neighs, size

    for(int i=1; i<=ntypes;i++) {
        unique_atomicNrs.push_back(getElementNumber(mass[i]));
    }
    for(int i=0; i<natoms; i++) {
        this->atomicNrs[i] = unique_atomicNrs[types[i]-1];
    }
    for(int i=0; i<natoms; i++) {
        this->num_neighs[i] = 0;
        // this->num_pair[i] = 0;
    }

    copy(unique_atomicNrs.begin(),unique_atomicNrs.end(),this->unique);
    for(int k=0; k < 3;k++) {
        for(int l=0; l < 3;l++) {
            this->box[3*k+l] = dbox[k][l];
        }
    }

    //get rid of lines 74-91 in calc_eon and can pass box and x directly
    for(int k=0; k < natoms;k++) {
        for(int l=0; l < 3;l++) {
            this->positions[3*k+l] = pos_car[k][l];
            this->pool_positions[3*k+l] = pos_car[k][l];
        }
    }
    this->bounds["x"].resize(2);
    this->bounds["x"][0] = bounds[0][0];//first is lo then is hi,?better way to organize??
    this->bounds["x"][1] = bounds[0][1];
    this->bounds["y"].resize(2);
    this->bounds["y"][0] = bounds[1][0];
    this->bounds["y"][1] = bounds[1][1];
    this->bounds["z"].resize(2);
    this->bounds["z"][0] = bounds[2][0];
    this->bounds["z"][1] = bounds[2][1];
    this->orig_bounds = this->bounds;//keep copy of original
    this->num_cells[0] = 1;
    this->num_cells[1] = 1;
    this->num_cells[2] = 1;
    this->is_orthogonal = ortho;
    setup();
}

void PyAMFFInterface::setup() {
    //should I put super declaration here as well
    if (new_pyamff == true){
        read_mlffParas(&this->num_atoms, &this->num_types, &this->max_fps, this->atomicNrs, this->unique);//dbl-chk num_types; used 2b num_elem
        prepfNN(&this->num_atoms, &this->num_types, &this->max_fps, this->atomicNrs, this->unique);
    }
    new_pyamff = false;
    // return;
}
//move if new_Pyamff loop so we can reset everything.
void PyAMFFInterface::reset(){
    for(int i=0; i < this->total_atoms; i++) {
        this->num_neighs[i] = 0;
        // this->num_pair[i] = 0;
        for(int j=0; j < this->max_neighs; j++) {
            this->neighs[i][j] = 0;
            this->adjust[i][j] = 0;
            // this->distances[i][j] = 0.0;
        }
    }
    this->noNeighs = true;
    // this->total_pairs = 0;
    //reset forces? energy?
    // cleanMemory();
    // setup();
}

void PyAMFFInterface::setMaxNeighs(int max) {
    if (max > 0) {
        this->max_neighs = max;
        if(!this->noNeighs) {
            for(int i=0; i < this->neighs.size(); i++) {
                this->neighs[i].resize(max);
                this->adjust[i].resize(max);
                // this->distances[i].resize(max);
            }
        }else {
            for(int i=0; i < this->neighs.size(); i++) {
                this->neighs[i] = vector<int>(max,0);
                this->adjust[i] = vector<int>(max,0);
                // this->distances[i] = vector<double>(max,0.0);
            }
        }
        
    }
    else {
        cout << "MAX NEIGHS MUST BE STRICTLY GREATER THAN ZERO" << endl;
        exit(2);
    }        
}

double PyAMFFInterface::getForcesEnergy(double* const* const& atomF) {
    int *neighlist;
    // double *distmat;
    //done this way so it will be exactly the same in fortran
    neighlist = new int[this->total_atoms*this->max_neighs];
    //sort then copy over to c/fortran-friendly array
    for (int i=0; i < this->total_atoms; i++) {
        //sort them then copy them over
        sort(this->neighs[i].begin(),this->neighs[i].begin()+this->num_neighs[i]);
        for(int j=0; j<this->max_neighs; j++) {
            neighlist[this->max_neighs*i+j] = this->neighs[i][j]+this->adjust[i][j];
        }
    }
    //definitely work for orthogonal, don't know about triclinic
    this->num_cells[0] = 2*ceil((orig_bounds["x"][0]-bounds["x"][0])/(orig_bounds["x"][1]-orig_bounds["x"][0])) +  1;
    this->num_cells[1] = 2*ceil((orig_bounds["y"][0]-bounds["y"][0])/(orig_bounds["y"][1]-orig_bounds["y"][0])) +  1;
    this->num_cells[2] = 2*ceil((orig_bounds["z"][0]-bounds["z"][0])/(orig_bounds["z"][1]-orig_bounds["z"][0])) +  1;
    int tncells = this->num_cells[0] * this->num_cells[1] * this->num_cells[2];
    //pyamff doesn't consider negative offset, should not add 1?
    setup();
    //can we convert on pyamff side
    //may need to reformat neighs and distances
    calc_lammps(&num_atoms, positions, box, atomicNrs, forces, &energy, &num_types, unique,&max_fps,&max_neighs,neighlist, &total_atoms, num_cells,pool_positions,neigh_tag); 
    //  # set atomF, note pyamff dus array, lammps does matrix
    for(int k=0; k < this->num_atoms;k++) {
        for(int l=0; l < 3;l++) {
            atomF[k][l] =forces[3*k+l];
        }
    }
    reset();
    return energy;
}

void PyAMFFInterface::addNeighbor(int i, int j) {
    // void PyAMFFInterface::addNeighbor(int i, int j,int tag) {
    //add to neighborlist, update distances, increment total_pairs, add to vector num_neighs
    this->neighs[i][this->num_neighs[i]] = j; 
    // this->neighs[j][this->num_neighs[j]] = i;
    //fortran is 0 based index, so I use and adjust so I don't have to change Lammps protocol
    this->adjust[i][this->num_neighs[i]] = 1; //should it be 1?
    // this->adjust[j][this->num_neighs[j]] = 1; 
    this->num_neighs[i] += 1;
    // this->num_neighs[j] += 1;
    if(!this->noNeighs) this->noNeighs = !this->noNeighs;
}

//make sure no exceed bounds
void PyAMFFInterface::updatePosition(int i, int tag, double* pos) {
    // void PyAMFFInterface::updatePosition(int i, double* pos) {
    for (int j=0; j < 3; j++) {
        if (i < this->num_atoms) {
            this->positions[3*i+j] = pos[j];
        }
        this->pool_positions[3*i+j] = pos[j];
    }
    //get mins and maxes
    if (pos[0] < this->bounds["x"][0]) {
        this->bounds["x"][0] = pos[0];
    }
    if (pos[1] < this->bounds["y"][0]) {
        this->bounds["y"][0] = pos[1];
    }
    if (pos[2] < this->bounds["z"][0]) {
        this->bounds["z"][0] = pos[2];
    }
    if (pos[0] > this->bounds["x"][1]) {
        this->bounds["x"][1] = pos[0];
    }
    if (pos[1] > this->bounds["y"][1]) {
        this->bounds["y"][1] = pos[1];
    }
    if (pos[2] > this->bounds["z"][1]) {
        this->bounds["z"][1] = pos[2];
    }
    this->neigh_tag[i] = tag;
}

PyAMFFInterface::element PyAMFFInterface::getElement(int number) {
    element elem;
    for(int i=0; i < 119; i++) {
        if (this->PeriodicTable[i].number == number) {
            return elem;
        }
    }
    //this should be an error statement
    cout << "UNABLE TO FIND ELEMENT WITH DESIRED MASS WITHIN GIVEN TOLERANCE" << endl;
    return this->PeriodicTable[0];
}
PyAMFFInterface::element PyAMFFInterface::getElement(double mass) {
    return getElement(mass, 1E-2);
}
PyAMFFInterface::element PyAMFFInterface::getElement(double mass, double tol) {
    double min = mass - tol;
    double max = mass + tol;
    element elem;
    for(int i=0; i < 119; i++) {
        elem = this->PeriodicTable[i];
        if (elem.mass <= max && elem.mass >= min) {
            return elem;
        }
    }
    //this should be an error statement
    cout << "UNABLE TO FIND ELEMENT WITH DESIRED MASS WITHIN GIVEN TOLERANCE" << endl;
    return this->PeriodicTable[0];
}

PyAMFFInterface::element PyAMFFInterface::getElement(std::string sym) {
    std::locale loc;
    element elem;

    cout << "WARNING: THIS WILL RETURN THE FIRST RESULT FOUND." << endl;
    for(int i=0; i < 119; i++) {
        elem = this->PeriodicTable[i];
        if (tolower(elem.symbol,loc).compare(tolower(sym,loc))) {
            return elem;
        }
    }
    cout << "UNABLE TO FIND ELEMENT WITH DESIRED MASS WITHIN GIVEN TOLERANCE" << endl;
    return this->PeriodicTable[0];
}

std::string PyAMFFInterface::getElementSymbol(int number) {
    return getElement(number).symbol;
}
std::string PyAMFFInterface::getElementSymbol(double mass){
    return getElement(mass).symbol;
}
std::string PyAMFFInterface::getElementSymbol(double mass, double tol){
    return getElement(mass,tol).symbol;
}
int PyAMFFInterface::getElementNumber(double mass){
    return getElement(mass).number;
}
int PyAMFFInterface::getElementNumber(double mass,double tol){
    return getElement(mass,tol).number;
}
int PyAMFFInterface::getElementNumber(std::string symbol){
    return getElement(symbol).number;
}
double PyAMFFInterface::getElementMass(int number){
    return getElement(number).mass;
}
double PyAMFFInterface::getElementMass(std::string symbol){
    return getElement(symbol).mass;
}


